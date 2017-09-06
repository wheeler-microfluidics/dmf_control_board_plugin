"""
Copyright 2011-2017 Ryan Fobel and Christian Fobel

This file is part of dmf_control_board.

dmf_control_board is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

dmf_control_board is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with dmf_control_board.  If not, see <http://www.gnu.org/licenses/>.
"""
from copy import deepcopy
import logging
import json
import math
import re
import warnings

from datetime import datetime
from dmf_control_board_firmware import (DMFControlBoard, FeedbackResultsSeries,
                                        feedback_results_to_impedance_frame)
from feedback import (FeedbackOptions, FeedbackOptionsController,
                      FeedbackCalibrationController, FeedbackResultsController,
                      RetryAction, SweepFrequencyAction, SweepVoltageAction)
from flatland import Integer, Boolean, Float, Form, Enum, String
from flatland.validation import ValueAtLeast, ValueAtMost
from microdrop.app_context import get_app, get_hub_uri
from microdrop.dmf_device import DeviceScaleNotSet
from microdrop.gui.protocol_grid_controller import ProtocolGridController
from microdrop.plugin_helpers import (StepOptionsController, AppDataController,
                                      get_plugin_info)
from microdrop.plugin_manager import (IPlugin, IWaveformGenerator, Plugin,
                                      implements, PluginGlobals,
                                      ScheduleRequest, emit_signal,
                                      get_service_instance,
                                      get_service_instance_by_name)
from microdrop_utility.gui import yesno, FormViewDialog
from nested_structures import apply_depth_first, apply_dict_depth_first
from path_helpers import path
from pygtkhelpers.gthreads import gtk_threadsafe
from pygtkhelpers.ui.dialogs import info as info_dialog
from zmq_plugin.plugin import Plugin as ZmqPlugin
from zmq_plugin.schema import decode_content_data
import arrow
import dmf_control_board_firmware as dmf
import gobject
import gtk
import microdrop_utility as utility
import numpy as np
import pandas as pd
import tables
import yaml
import zmq

from ._version import get_versions
from .wizards import MicrodropChannelsAssistantView

__version__ = get_versions()['version']
del get_versions

logger = logging.getLogger(__name__)

# Ignore natural name warnings from PyTables [1].
#
# [1]: https://www.mail-archive.com/pytables-users@lists.sourceforge.net/msg01130.html
warnings.simplefilter('ignore', tables.NaturalNameWarning)

PluginGlobals.push_env('microdrop.managed')


class DmfZmqPlugin(ZmqPlugin):
    '''
    API for adding/clearing droplet routes.
    '''
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        self._electrode_commands_registered = 0
        super(DmfZmqPlugin, self).__init__(*args, **kwargs)

    def check_sockets(self):
        '''
        Check for messages on command and subscription sockets and process
        any messages accordingly.
        '''
        try:
            msg_frames = self.command_socket.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            pass
        else:
            self.on_command_recv(msg_frames)

        try:
            msg_frames = self.subscribe_socket.recv_multipart(zmq.NOBLOCK)
            source, target, msg_type, msg_json = msg_frames
            if all([source == 'microdrop.electrode_controller_plugin',
                    msg_type == 'execute_reply']):
                # The 'electrode_controller_plugin' plugin maintains the
                # requested state of each electrode.
                msg = json.loads(msg_json)
                if msg['content']['command'] in ('set_electrode_state',
                                                 'set_electrode_states'):
                    data = decode_content_data(msg)
                    self.parent.actuated_area = data['actuated_area']
                    self.parent.update_channel_states(data['channel_states'])
                elif msg['content']['command'] == 'get_channel_states':
                    data = decode_content_data(msg)
                    self.parent.actuated_area = data['actuated_area']
                    self.parent.channel_states = \
                        self.parent.channel_states.iloc[0:0]
                    self.parent.update_channel_states(data['channel_states'])
            elif (self._electrode_commands_registered < 2 and
                  (source == 'dmf_device_ui_plugin')):
                # Register electrode commands with device UI plugin.
                logger.info('Register electrode commands with device UI '
                            'plugin.')
                for title, command in (('Measure capacitance of liquid',
                                        'measure_cap_liquid'),
                                       ('Measure capacitance of filler media',
                                        'measure_cap_filler')):
                    def on_registered(reply):
                        self._electrode_commands_registered += 1
                    self.execute_async('dmf_device_ui_plugin',
                                       'register_electrode_command',
                                       extra_kwargs={'command': command},
                                       title=title, callback=on_registered)
            else:
                self.most_recent = msg_json
        except zmq.Again:
            pass
        except Exception:
            logger.error('Error processing message from subscription '
                         'socket.', exc_info=True)
        return True

    def on_execute__channel_count(self, request):
        return self.parent.control_board.number_of_channels()

    def on_execute__measure_impedance(self, request):
        '''
        Measure impedance while the channels specified by `state` field are
        actuated (no actuated channels by default).
        '''
        data = decode_content_data(request)
        control_board = self.parent.control_board

        n_sampling_windows = data.pop('n_sampling_windows')
        feedback_results = self.measure(control_board.measure_impedance,
                                        n_sampling_windows, **data)
        return feedback_results_to_impedance_frame(feedback_results)

    def on_execute__sweep_channels(self, request):
        '''
        Measure impedance while the channels specified by `state` field are
        actuated (no actuated channels by default).
        '''
        data = decode_content_data(request)
        control_board = self.parent.control_board
        n_sampling_windows = data.pop('n_sampling_windows')
        df_impedances = self.measure(control_board.sweep_channels,
                                     n_sampling_windows, **data)
        return df_impedances

    def on_execute__measure_cap_filler(self, request):
        '''
        Measure capacitance of actuated electrodes as filler (e.g., air, oil).
        '''
        c = self.parent.feedback_options_controller.measure_cap_filler()
        logger.info('[measure_cap_filler] c=%s', c)
        return c

    def on_execute__measure_cap_liquid(self, request):
        '''
        Measure capacitance of actuated electrodes as liquid (i.e., droplet).
        '''
        c = self.parent.feedback_options_controller.measure_cap_liquid()
        logger.info('[measure_cap_liquid] c=%s', c)
        return c

    def measure(self, measure_func, n_sampling_windows, **kwargs):
        '''
        Measure impedance while the channels specified by `state` field are
        actuated (no actuated channels by default).
        '''
        control_board = self.parent.control_board

        if 'voltage' in kwargs:
            start_voltage = control_board.waveform_voltage()
            control_board.set_waveform_voltage(kwargs['voltage'])
        if 'frequency' in kwargs:
            control_board.set_waveform_frequency(kwargs['frequency'])
            start_frequency = control_board.waveform_frequency()

        try:
            app_values = self.parent.get_app_values()

            # Set unspecified measurement parameters to plugin app option
            # values.
            sampling_window_ms = kwargs.get('sampling_window_ms',
                                            app_values['sampling_window_ms'])
            delay_between_windows_ms = kwargs.get('delay_between_windows_ms',
                                                  app_values
                                                  ['delay_between_windows_ms'])
            interleave_samples = kwargs.get('interleave_samples', app_values
                                            ['interleave_feedback_samples'])
            use_rms = kwargs.get('use_rms', app_values['use_rms'])

            channel_count = control_board.number_of_channels()
            state = kwargs.get('state', channel_count * [0])
            return measure_func(sampling_window_ms,
                                n_sampling_windows,
                                delay_between_windows_ms, interleave_samples,
                                use_rms, state)
        finally:
            # Restore original voltage and frequency as required.
            if 'voltage' in kwargs:
                control_board.set_waveform_voltage(start_voltage)
            if 'frequency' in kwargs:
                control_board.set_waveform_frequency(start_frequency)


def microdrop_experiment_log_to_feedback_results_df(log):
    # get the FeedbackResults object for each step
    results = log.get('FeedbackResults',
                      plugin_name=get_plugin_info(path(__file__).parent)
                      .plugin_name)

    # Get the start time for each step (in seconds), relative to the beginning
    # of the protocol.
    step_start_time = log.get('time')

    start_time = arrow.get(log.get('start time')[0])

    # combine all steps in the protocol into a single dataframe
    feedback_results_df = pd.DataFrame()
    for i, step in enumerate(results):
        if step is None:
            continue
        else:
            # reset index (step_time is no longer unique for the full dataset)
            df = step.to_frame().reset_index()

            # add step_index and utc_time columns (index by utc_time)
            df.insert(0, 'step_index', i)
            df.insert(0, 'utc_timestamp', [start_time.replace(
                        seconds=step_start_time[i] + t).datetime
                                      for t in df['step_time']])
            df.set_index('utc_timestamp', inplace=True)

            feedback_results_df = feedback_results_df.append(df)
    return feedback_results_df


def feedback_results_df_to_step_summary_df(feedback_results_df):
    if len(feedback_results_df) == 0:
        # return an empty dataframe
        return pd.DataFrame()

    # create a step summary dataframe for the experiment
    grouped = feedback_results_df.reset_index().groupby('step_index')

    # by default, use the last value in the group
    # (e.g., final capacitance, Z_device)
    steps = grouped.last()

    # for utc_time, use the first value in the group (i.e., start of the step)
    steps[['utc_timestamp']] = grouped.first()[['utc_timestamp']]

    # for force and voltage, use the mean value
    steps[['force', 'voltage']] = grouped.mean()[['force', 'voltage']]

    # remove dxdt and dxdt_filtered
    del steps['dxdt'], steps['dxdt_filtered']

    # rename columns
    steps.rename(columns={'step_time': 'step_duration',
                          'force': 'mean_force',
                          'voltage': 'mean_voltage',
                          'Z_device_filtered': 'final_Z_device_filtered',
                          'capacitance_filtered': 'final_capacitance_filtered',
                          'x_position_filtered': 'final_x_position_filtered',
                          'Z_device': 'final_Z_device',
                          'capacitance': 'final_capacitance',
                          'x_position': 'final_x_position'},
                 inplace=True)

    return steps.reset_index().set_index('utc_timestamp')


class DMFControlBoardOptions(object):
    _default_force = 25.0

    def __init__(self, duration=100, voltage=100.0, frequency=10e3,
                 feedback_options=None, force=None):
        self.duration = duration
        if feedback_options is None:
            self.feedback_options = FeedbackOptions()
        else:
            self.feedback_options = feedback_options
        if force is None:
            force = self._default_force
        self.voltage = voltage
        self.frequency = frequency
        self.force = force

    def __setstate__(self, state):
        self.__dict__ = state
        self._upgrade()

    def _upgrade(self):
        """
        Upgrade the serialized object if necessary.
        """
        if not hasattr(self, 'force'):
            self.force = self._default_force

    def to_dict(self):
        result = {k: getattr(self, k) for k in ('duration', 'force',
                                                'frequency', 'voltage')}
        result['feedback_options'] = self.feedback_options.to_dict()
        result['__class__'] = '.'.join([self.__class__.__module__,
                                        self.__class__.__name__])
        return result

    @classmethod
    def from_dict(self, options_dict):
        options_dict.pop('__class__', None)
        feedback_options = options_dict.pop('feedback_options')
        options_dict['feedback_options'] = (FeedbackOptions
                                            .from_dict(feedback_options))
        return DMFControlBoardOptions(**options_dict)


def format_func(value):
    if value:
        return True
    else:
        return False


def max_voltage(element, state):
    """Verify that the voltage is below a set maximum"""
    service = get_service_instance_by_name(
        get_plugin_info(path(__file__).parent).plugin_name)

    if service.control_board.connected() and (element.value >
                                              service.control_board
                                              .max_waveform_voltage):
        return element.errors.append('Voltage exceeds the maximum value '
                                     '(%d V).' %
                                     service.control_board.max_waveform_voltage)
    else:
        return True


def check_frequency(element, state):
    """Verify that the frequency is within the valid range"""
    service = get_service_instance_by_name(
        get_plugin_info(path(__file__).parent).plugin_name)

    if service.control_board.connected() and (element.value <
                                              service.control_board
                                              .min_waveform_frequency
                                              or element.value >
                                              service.control_board
                                              .max_waveform_frequency):
        return element.errors.append('Frequency is outside of the valid range '
                                     '(%.1f - %.1f Hz).' %
                                     (service.control_board
                                      .min_waveform_frequency,
                                      service.control_board
                                      .max_waveform_frequency))
    else:
        return True


class DMFControlBoardPlugin(Plugin, StepOptionsController, AppDataController):
    """
    This class is automatically registered with the PluginManager.
    """
    implements(IPlugin)
    implements(IWaveformGenerator)

    @property
    def AppFields(self):
        # Get list of ports matching Mega2560 USB vendor/product ID.
        comports = dmf.serial_ports().index.tolist()
        default_port = comports[0] if comports else None
        return Form.of(Integer.named('sampling_window_ms')
                       .using(default=5, optional=True,
                              validators=[ValueAtLeast(minimum=0)]),
                       Integer.named('delay_between_windows_ms')
                       .using(default=0, optional=True,
                              validators=[ValueAtLeast(minimum=0)]),
                       Boolean.named('use_rms').using(default=True,
                                                      optional=True),
                       Boolean.named('interleave_feedback_samples')
                       .using(default=True, optional=True),
                       Enum.named('serial_port').using(default=default_port,
                                                       optional=True)
                       .valued(*comports),
                       Integer.named('baud_rate')
                       .using(default=115200, optional=True,
                              validators=[ValueAtLeast(minimum=0)]),
                       Boolean.named('auto_atx_power_off')
                       .using(default=False, optional=True),
                       Boolean.named('use_force_normalization')
                       .using(default=False, optional=True),
                       String.named('c_drop').using(default='', optional=True,
                                                    properties={'show_in_gui':
                                                                False}),
                       String.named('c_filler')
                       .using(default='', optional=True,
                              properties={'show_in_gui': False}))

    StepFields = Form.of(
        Integer.named('duration').using(default=100, optional=True,
                                        validators=[ValueAtLeast(minimum=0)]),
        Float.named('voltage').using(default=100, optional=True,
                                     validators=[ValueAtLeast(minimum=0),
                                                 max_voltage]),
        Float.named('force').using(default=25, optional=True,
                                   validators=[ValueAtLeast(minimum=0)]),
        Float.named('frequency').using(default=10e3, optional=True,
                                       validators=[ValueAtLeast(minimum=0),
                                                   check_frequency]),
        Boolean.named('feedback_enabled').using(default=True, optional=True),
    )
    _feedback_fields = set(['feedback_enabled'])

    version = __version__

    def __init__(self):
        '''
        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code blocks,
            ensuring the code runs in the main GTK thread.

        .. versionchanged:: 2.3.4
            Use :data:`__version__` for plugin version.
        '''
        self.control_board = DMFControlBoard()
        self.name = get_plugin_info(path(__file__).parent).plugin_name
        self.url = self.control_board.host_url()
        self.steps = []  # list of steps in the protocol
        self.feedback_options_controller = None
        self.feedback_results_controller = None
        self.feedback_calibration_controller = None
        self.initialized = False
        self.connection_status = "Not connected"
        self.n_voltage_adjustments = None
        self.amplifier_gain_initialized = False
        self.current_frequency = None

        self.timeout_id = None
        self.watchdog_timeout_id = None
        self.menu_actions = ['Test channels...',
                             ('Calibration',
                              ['Calibrate reference load',
                               'Open reference load calibration',
                               'Calibrate device load',
                               'Open device load calibration']),
                             ('Configuration',
                              ['Reset to default values',
                               'Edit settings',
                               'Load from file',
                               'Save to file'])]
        self.actuated_area = 0
        self.channel_states = pd.Series()
        self.plugin = None
        self.plugin_timeout_id = None

        @gtk_threadsafe
        def _init_menu_ui():
            self.save_control_board_configuration = \
                gtk.MenuItem("Edit configuration")
            self.edit_log_calibration_menu_item = gtk.MenuItem("Edit "
                                                               "calibration")
            self.save_log_calibration_menu_item = gtk.MenuItem("Save "
                                                               "calibration "
                                                               "to file")
            self.load_log_calibration_menu_item = gtk.MenuItem("Load "
                                                               "calibration "
                                                               "from file")
        _init_menu_ui()

    def update_channel_states(self, channel_states):
        # Update locally cached channel states with new modified states.
        try:
            self.channel_states = channel_states.combine_first(self
                                                               .channel_states)
        except ValueError:
            logging.info('channel_states: %s', channel_states)
            logging.info('self.channel_states: %s', self.channel_states)
            logging.info('', exc_info=True)
        else:
            app = get_app()
            if self.control_board.connected() and (app.realtime_mode or
                                                   app.running):
                self.on_step_run()

    def cleanup_plugin(self):
        if self.plugin_timeout_id is not None:
            gobject.source_remove(self.plugin_timeout_id)
        if self.plugin is not None:
            self.plugin = None

    def on_plugin_enable(self):
        '''
        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code blocks,
            ensuring the code runs in the main GTK thread.
        '''
        logger.info('on_plugin_enable')
        super(DMFControlBoardPlugin, self).on_plugin_enable()

        self.cleanup_plugin()
        # Initialize 0MQ hub plugin and subscribe to hub messages.
        self.plugin = DmfZmqPlugin(self, self.name, get_hub_uri(),
                                   subscribe_options={zmq.SUBSCRIBE: ''})
        # Initialize sockets.
        self.plugin.reset()

        # Periodically process outstanding message received on plugin sockets.
        self.plugin_timeout_id = gtk.timeout_add(10, self.plugin.check_sockets)

        @gtk_threadsafe
        def _init_ui():
            self.feedback_options_controller = FeedbackOptionsController(self)
            self.feedback_results_controller = FeedbackResultsController(self)
            self.feedback_calibration_controller = (
                FeedbackCalibrationController(self))
            self.edit_log_calibration_menu_item.connect(
                "activate",
                self.feedback_calibration_controller.on_edit_log_calibration)
            self.save_log_calibration_menu_item.connect(
                "activate",
                self.feedback_calibration_controller.on_save_log_calibration)
            self.load_log_calibration_menu_item.connect(
                "activate",
                self.feedback_calibration_controller.on_load_log_calibration)

            experiment_log_controller = get_service_instance_by_name(
                "microdrop.gui.experiment_log_controller", "microdrop")

            if hasattr(experiment_log_controller, 'popup'):
                experiment_log_controller.popup.add_item(
                    self.edit_log_calibration_menu_item)
                experiment_log_controller.popup.add_item(
                    self.save_log_calibration_menu_item)
                experiment_log_controller.popup.add_item(
                    self.load_log_calibration_menu_item)

            app = get_app()

            self.control_board_menu_item = gtk.MenuItem("DMF control board")
            app.main_window_controller.menu_tools.append(
                self.control_board_menu_item)

            self.control_board_menu = gtk.Menu()
            self.control_board_menu.show()
            self.control_board_menu_item.set_submenu(self.control_board_menu)

            self.feedback_options_controller.on_plugin_enable()

            def prepare_menu_item(node, parents, children, *args):
                menu_item = gtk.MenuItem(node)
                menu_item.show()

                if children:
                    # The node has children, so we need to create a GTK menu to
                    # hold the child menu items. We also must create a menu item
                    # for the label of the sub-menu.
                    menu = gtk.Menu()
                    menu_item.set_submenu(menu)
                    menu.show()
                    return (menu_item, menu)
                else:
                    return (menu_item, None)

            def attach_menu_item(key, node, parents, *args):
                if parents:
                    # Extract menu item of nearest parent.
                    parent_item = parents[-1][1].item[1]
                else:
                    # Use main plugin menu as parent.
                    parent_item = self.control_board_menu
                parent_item.append(node.item[0])
                node.item[0].show()

            # Prepare menu items for layout defined in `self.menu_actions`.
            self.menu_items = apply_depth_first(self.menu_actions,
                                                as_dict=True,
                                                func=prepare_menu_item)
            # Attach each menu item to the corresponding parent menu.
            apply_dict_depth_first(self.menu_items, attach_menu_item)

            @gtk_threadsafe
            def test_channels(*args):
                '''
                Launch wizard to test actuation of a bank of switches *(i.e.,
                all switches on a single switching board)*.

                Append results to an [HDF][1] file, where measurements from the
                same run share a common value in the `timestamp` column.
                '''
                view = MicrodropChannelsAssistantView(self.control_board)

                def on_close(*args):
                    view.to_hdf(self.calibrations_dir()
                                .joinpath('[%05d]-channels.h5' %
                                          self.control_board.serial_number))
                view.widget.connect('close', on_close)
                view.show()

            # Connect the action for each menu item to the corresponding
            # call-back function.
            self.menu_items['Test channels...'].item[0].connect('activate',
                                                                test_channels)
            menu = self.menu_items['Configuration']
            menu['Edit settings'][0].connect('activate',
                                             self.on_edit_configuration)
            menu['Save to file'][0].connect('activate',
                                            lambda *a:
                                            self.save_config_dialog())
            menu['Load from file'][0].connect('activate',
                                              lambda *a:
                                              self.load_config_dialog())
            menu['Reset to default values'][0].connect(
                'activate', self.on_reset_configuration_to_default_values)

            menu = self.menu_items['Calibration']
            menu['Calibrate reference load'][0].connect(
                'activate',
                self.feedback_calibration_controller.on_perform_calibration)
            menu['Open reference load calibration'][0].connect(
                'activate',
                lambda *args:
                self.feedback_calibration_controller
                .load_reference_calibration())
            menu['Calibrate device load'][0].connect(
                'activate',
                lambda *args: self.feedback_calibration_controller
                .calibrate_impedance())
            menu['Open device load calibration'][0].connect(
                'activate',
                lambda *args:
                self.feedback_calibration_controller
                .load_impedance_calibration())

            self.initialized = True

        if not self.initialized:
            _init_ui()

        self.check_device_name_and_version()

        @gtk_threadsafe
        def _refresh_ui():
            self.control_board_menu_item.show()
            self.edit_log_calibration_menu_item.show()
            self.feedback_results_controller.feedback_results_menu_item.show()

        _refresh_ui()

        if get_app().protocol:
            self.on_step_run()
            self._update_protocol_grid()

    def on_plugin_disable(self):
        self.cleanup_plugin()
        self.feedback_options_controller.on_plugin_disable()

        @gtk_threadsafe
        def _refresh_ui():
            self.control_board_menu_item.hide()
            self.edit_log_calibration_menu_item.hide()
            self.feedback_results_controller.window.hide()
            self.feedback_results_controller.feedback_results_menu_item.hide()

        _refresh_ui()

        if get_app().protocol:
            self.on_step_run()
            self._update_protocol_grid()

    def on_app_exit(self):
        """
        Handler called just before the Microdrop application exits.
        """
        self.cleanup_plugin()

    def on_protocol_swapped(self, old_protocol, protocol):
        self._update_protocol_grid()

    @gtk_threadsafe
    def _update_protocol_grid(self):
        app = get_app()
        app_values = self.get_app_values()
        pgc = get_service_instance(ProtocolGridController, env='microdrop')
        if pgc.enabled_fields:
            if self.name in pgc.enabled_fields.keys():
                if app_values['use_force_normalization']:
                    if 'voltage' in pgc.enabled_fields[self.name]:
                        pgc.enabled_fields[self.name].remove('voltage')
                    if 'force' not in pgc.enabled_fields[self.name]:
                        pgc.enabled_fields[self.name].add('force')

                    if app.protocol and (self.control_board.calibration is not
                                         None and self.control_board
                                         .calibration._c_drop):
                        for i, step in enumerate(app.protocol):
                            options = self.get_step_options(i)
                            options.voltage = \
                                self.control_board.force_to_voltage(
                                    options.force, options.frequency)
                else:
                    if 'force' in pgc.enabled_fields[self.name]:
                        pgc.enabled_fields[self.name].remove('force')
                    if 'voltage' not in pgc.enabled_fields[self.name]:
                        pgc.enabled_fields[self.name].add('voltage')

                pgc.update_grid()

    def on_app_options_changed(self, plugin_name):
        '''
        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code, ensuring the
            code runs in the main GTK thread.
        '''
        @gtk_threadsafe
        def _cached_capacitance_prompt_and_serial_settings():
            app_values = self.get_app_values()
            reconnect = False

            if self.control_board.connected():
                if 'c_drop' in app_values and (self.control_board.calibration
                                               ._c_drop is None):
                    c_drop = yaml.load(app_values['c_drop'])
                    if c_drop is not None and len(c_drop):
                        response = yesno('Use cached value for'
                                         'c<sub>drop</sub>?')
                        if response == gtk.RESPONSE_YES:
                            self.control_board.calibration._c_drop = c_drop
                        else:
                            self.set_app_values(dict(c_drop=''))
                if 'c_filler' in app_values and (self.control_board.calibration
                                                 ._c_filler is None):
                    c_filler = yaml.load(app_values['c_filler'])
                    if c_filler is not None and len(c_filler):
                        response = yesno('Use cached value for '
                                         'c<sub>filler</sub>?')
                        if response == gtk.RESPONSE_YES:
                            self.control_board.calibration._c_filler = c_filler
                        else:
                            self.set_app_values(dict(c_filler=''))

                if self.control_board.baud_rate != app_values['baud_rate']:
                    self.control_board.baud_rate = app_values['baud_rate']
                    reconnect = True
                if self.control_board.port != app_values['serial_port']:
                    reconnect = True

                if not reconnect:
                    # We're not reconnecting.  Update the watchdog timer.
                    self._update_watchdog(app_values['auto_atx_power_off'])

                if reconnect:
                    self.connect()
            self._update_protocol_grid()

        app = get_app()

        if plugin_name == self.name:
            _cached_capacitance_prompt_and_serial_settings()
        elif plugin_name == app.name:
            if self.control_board.connected() and (not app.realtime_mode and
                                                   not app.running):
                # We're not in realtime mode and not running a protocol.
                # Turn off all electrodes.
                logger.info('Turning off all electrodes.')
                self.control_board.set_state_of_all_channels(
                    np.zeros(self.control_board.number_of_channels()))
        if self.feedback_options_controller:
            (self.feedback_options_controller
             .on_app_options_changed(plugin_name))

    def connect(self):
        '''
        Try to connect to the control board at the default serial port selected
        in the Microdrop application options.

        If unsuccessful, try to connect to the control board on any available
        serial port, one-by-one.
        '''
        self.current_frequency = None
        self.amplifier_gain_initialized = False
        # Get list of Mega2560 serial ports.
        comports = dmf.serial_ports().index.tolist()
        if len(comports):
            # Try to connect to the last successful port (if it is available).
            app_values = self.get_app_values()
            most_recent_port = str(app_values['serial_port'])
            if most_recent_port in comports:
                # Most recently connected port is available.  Move it to the
                # head of the list of COM ports to try to connect on.
                comports.remove(most_recent_port)
                comports.insert(0, most_recent_port)
            elif most_recent_port:
                logger.warning('Control board not found on most recently '
                               'connected port (%s). Checking other ports...',
                               most_recent_port)
            # Try to connect to control board on available ports.
            self.control_board.connect(comports, app_values['baud_rate'])
            app_values['serial_port'] = self.control_board.port
            self.set_app_values(app_values)
        else:
            raise Exception("No serial ports available.")
        self._update_watchdog(app_values['auto_atx_power_off'])

    def _update_watchdog(self, enabled):
        try:
            if enabled:
                # Try to enable watchdog-timer to shut off power supply when
                # the `Microdrop` app is closed.
                self.control_board.watchdog_state = True
                self.control_board.watchdog_enabled = True
                if self.watchdog_timeout_id is None:
                    self.watchdog_timeout_id = gobject.timeout_add(
                        2000,  # Trigger every 2 seconds.
                        self._callback_reset_watchdog)
            else:
                # Try to disable the watchdog-timer
                self.control_board.watchdog_enabled = False
                # Kill any running timer
                if self.watchdog_timeout_id:
                    gobject.source_remove(self.watchdog_timeout_id)
                    self.watchdog_timeout_id = None
        except Exception:
            # earlier versions of the firmware may not accept this command,
            # so we need to catch any exceptions
            pass

    def _callback_reset_watchdog(self):
        # only reset the watchdog if we are connected and not waiting for a
        # reply
        if self.control_board.connected():
            waiting_for_reply = self.control_board.waiting_for_reply()
            if not waiting_for_reply:
                logger.debug('Reset watchdog')
                self.control_board.watchdog_state = True
            else:
                logger.debug("Don't reset watchdog. Waiting for reply to"
                             " previous command.")
        # [Return `True`][1] to request to be called again.
        #
        # [1]: http://www.pygtk.org/pygtk2reference/gobject-functions.html#function-gobject--timeout-add
        return True

    def check_device_name_and_version(self):
        '''
        Check to see if:

         a) The connected device is a DMF controller
         b) The device firmware matches the host driver API version

        In the case where the device firmware version does not match, display a
        dialog offering to flash the device with the firmware version that
        matches the host driver API version.

        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code, ensuring the
            code runs in the main GTK thread.
        '''
        try:
            self.connect()
            name = self.control_board.name()
            if name != "Arduino DMF Controller":
                raise Exception("Device is not an Arduino DMF Controller")

            host_software_version = self.control_board.host_software_version()
            remote_software_version = self.control_board.software_version()

            @gtk_threadsafe
            def _firmware_update_prompt():
                response = yesno("The control board firmware version (%s) "
                                 "does not match the driver version (%s). "
                                 "Update firmware?" % (remote_software_version,
                                                       host_software_version))
                if response == gtk.RESPONSE_YES:
                    self.on_flash_firmware()

            # Reflash the firmware if it is not the right version.
            if host_software_version != remote_software_version:
                _firmware_update_prompt()
        except Exception, why:
            logger.warning("%s" % why)

        self.update_connection_status()

    def on_flash_firmware(self, widget=None, data=None):
        '''
        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code, ensuring the
            code runs in the main GTK thread.
        '''
        @gtk_threadsafe
        def _config_save_prompt_and_flash():
            response = yesno("Save current control board configuration before "
                             "flashing?")
            if response == gtk.RESPONSE_YES:
                self.save_config_dialog()
            try:
                hardware_version =\
                    utility.Version.fromstring(self.control_board
                                               .hardware_version())
                if not connected:
                    self.control_board.disconnect()
                self.control_board.flash_firmware(hardware_version)
                app.main_window_controller.info("Firmware updated "
                                                "successfully.",
                                                "Firmware update")
            except Exception, why:
                logger.error("Problem flashing firmware. ""%s", why)
            self.check_device_name_and_version()

        app = get_app()
        connected = self.control_board.connected()
        if not connected:
            self.connect()
        _config_save_prompt_and_flash()

    def load_config_dialog(self):
        '''
        Load control-board device configuration from file, including values set
        during calibration_, and write the configuration to the control
        board.

        .. note::

        The behaviour of this method is described in `ticket #41`__.

        .. versionchanged:: 2.3.3
            Rename from :meth:`load_config` to :meth:`load_config_dialog` to
            emphasize that this method includes GTK code and **MUST** be
            executed within the main GTK thread.

        __ http://microfluidics.utoronto.ca/trac/dropbot/ticket/41
        .. _calibration: http://microfluidics.utoronto.ca/trac/dropbot/wiki/Control%20board%20calibration
        '''
        dialog = gtk.FileChooserDialog(
            title="Load control board configuration from file",
            action=gtk.FILE_CHOOSER_ACTION_OPEN,
            buttons=(gtk.STOCK_CANCEL,
                     gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN,
                     gtk.RESPONSE_OK)
        )
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_current_folder(self.configurations_dir())
        response = dialog.run()
        filename = path(dialog.get_filename())
        dialog.destroy()

        if response == gtk.RESPONSE_OK:
            try:
                config = yaml.load(filename.bytes())
            except Exception:
                logger.error('Error parsing control-board configuration '
                             'file.\n\n'
                             'Please ensure the configuration file is a valid'
                             'YAML-encoded file.')
            else:
                self.control_board.write_config(config)
                message = ('Successfully wrote persistent configuration '
                           'settings to control-board.')
                logger.info(message)
                info_dialog(message)

    def save_config_dialog(self):
        '''
        Save control-board device configuration, including values set during
        calibration_.

        .. note::
            The behaviour of this method is described in `ticket #41`__.

        .. versionchanged:: 2.3.3
            Rename from :meth:`save_config` to :meth:`save_config_dialog` to
            emphasize that this method includes GTK code and **MUST** be
            executed within the main GTK thread.

        __ http://microfluidics.utoronto.ca/trac/dropbot/ticket/41
        .. _calibration: http://microfluidics.utoronto.ca/trac/dropbot/wiki/Control%20board%20calibration
        '''
        dialog = gtk.FileChooserDialog(
            title="Save control board configuration to file",
            action=gtk.FILE_CHOOSER_ACTION_SAVE,
            buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN,
                     gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_current_folder(self.configurations_dir())
        dialog.set_current_name(self._file_prefix() + 'config.yml')
        response = dialog.run()
        filename = path(dialog.get_filename())
        dialog.destroy()

        if response == gtk.RESPONSE_OK:
            if filename.isfile():
                response = yesno('File exists. Would you like to overwrite '
                                 'it?')
                if response != gtk.RESPONSE_YES:
                    return
            self.to_yaml(filename)

    def to_yaml(self, output_path):
        '''
        Write control board configuration to a YAML output file.
        '''
        config = self.control_board.read_config()
        config = dict([(k, v) for k, v in config.iteritems() if v is not None])
        for k in config.keys():
            # Yaml doesn't support serializing numpy scalar types, but the
            # configuration returned by `read_config` may contain numpy
            # floating point values.  Therefore, we check and cast each
            # numpy float as a native Python float.
            if isinstance(config[k], np.float32):
                config[k] = float(config[k])
        config_str = yaml.dump(config)
        with open(output_path, 'wb') as output:
            print >> output, '''
# DropBot DMF control-board configuration
# =======================================
#'
# This file contains the configuration [settings][1] for the control-board in a
# [DropBot][2] [digital-microfluidics][3] system.
#
# [1]: http://microfluidics.utoronto.ca/trac/dropbot/ticket/41#ticket
# [2]: http://microfluidics.utoronto.ca/trac/dropbot
# [3]: http://microfluidics.utoronto.ca'''.strip()
            print >> output, config_str

    def on_edit_configuration(self, widget=None, data=None):
        '''
        Display a dialog to manually edit the configuration settings for the
        control board.  These settings include values that are automatically
        adjusted during calibration_.

        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code, ensuring the
            code runs in the main GTK thread.

        .. _calibration: http://microfluidics.utoronto.ca/trac/dropbot/wiki/Control%20board%20calibration
        '''
        if not self.control_board.connected():
            logger.error("A control board must be connected in order to "
                         "edit configuration settings.")
            return

        hardware_version = utility.Version.fromstring(
            self.control_board.hardware_version())

        schema_entries = []
        settings = {}
        settings['amplifier_gain'] = self.control_board.amplifier_gain
        schema_entries.append(
            Float.named('amplifier_gain')
            .using(default=settings['amplifier_gain'], optional=True,
                   validators=[ValueAtLeast(minimum=0.01)]))
        settings['auto_adjust_amplifier_gain'] = \
            self.control_board.auto_adjust_amplifier_gain
        schema_entries.append(
            Boolean.named('auto_adjust_amplifier_gain')
            .using(default=settings['auto_adjust_amplifier_gain'],
                   optional=True))
        settings['voltage_tolerance'] = self.control_board.voltage_tolerance
        schema_entries.append(
            Float.named('voltage_tolerance')
            .using(default=settings['voltage_tolerance'], optional=True,
                   validators=[ValueAtLeast(minimum=0)]))
        settings['use_antialiasing_filter'] = \
            self.control_board.use_antialiasing_filter
        schema_entries.append(
            Boolean.named('use_antialiasing_filter')
            .using(default=settings['use_antialiasing_filter'], optional=True))
        settings['max_waveform_voltage'] = \
            self.control_board.max_waveform_voltage
        schema_entries.append(
            Float.named('max_waveform_voltage')
            .using(default=settings['max_waveform_voltage'], optional=True,
                   validators=[ValueAtLeast(minimum=0)]))
        settings['min_waveform_frequency'] = \
            self.control_board.min_waveform_frequency
        schema_entries.append(
            Float.named('min_waveform_frequency')
            .using(default=settings['min_waveform_frequency'], optional=True,
                   validators=[ValueAtLeast(minimum=0)]))
        settings['max_waveform_frequency'] = \
            self.control_board.max_waveform_frequency
        schema_entries.append(
            Float.named('max_waveform_frequency')
            .using(default=settings['max_waveform_frequency'], optional=True,
                   validators=[ValueAtLeast(minimum=0)]))
        if hardware_version.major == 1:
            settings['WAVEOUT_GAIN_1'] = self.control_board.waveout_gain_1
            schema_entries.append(
                Integer.named('WAVEOUT_GAIN_1')
                .using(default=settings['WAVEOUT_GAIN_1'], optional=True,
                       validators=[ValueAtLeast(minimum=0),
                                   ValueAtMost(maximum=255)]))
            settings['VGND'] = self.control_board.vgnd
            schema_entries.append(
                Integer.named('VGND')
                .using(default=settings['VGND'], optional=True,
                       validators=[ValueAtLeast(minimum=0),
                                   ValueAtMost(maximum=255)]))
        else:
            settings['SWITCHING_BOARD_I2C_ADDRESS'] = \
                self.control_board.switching_board_i2c_address
            schema_entries.append(
                Integer.named('SWITCHING_BOARD_I2C_ADDRESS')
                .using(default=settings['SWITCHING_BOARD_I2C_ADDRESS'],
                       optional=True, validators=[ValueAtLeast(minimum=0),
                                                  ValueAtMost(maximum=255)]))
            settings['SIGNAL_GENERATOR_BOARD_I2C_ADDRESS'] = (
                self.control_board.signal_generator_board_i2c_address)
            schema_entries.append(
                Integer.named('SIGNAL_GENERATOR_BOARD_I2C_ADDRESS')
                .using(default=settings['SIGNAL_GENERATOR_BOARD_I2C_ADDRESS'],
                       optional=True, validators=[ValueAtLeast(minimum=0),
                                                  ValueAtMost(maximum=255)]))
        for i in range(len(self.control_board.calibration.R_hv)):
            settings['R_hv_%d' % i] = self.control_board.calibration.R_hv[i]
            schema_entries.append(
                Float.named('R_hv_%d' % i)
                .using(default=settings['R_hv_%d' % i], optional=True,
                       validators=[ValueAtLeast(minimum=0)]))
            settings['C_hv_%d' % i] = (self.control_board.calibration.C_hv[i] *
                                       1e12)
            schema_entries.append(
                Float.named('C_hv_%d' % i)
                .using(default=settings['C_hv_%d' % i], optional=True,
                       validators=[ValueAtLeast(minimum=0)]))
        for i in range(len(self.control_board.calibration.R_fb)):
            settings['R_fb_%d' % i] = self.control_board.calibration.R_fb[i]
            schema_entries.append(
                Float.named('R_fb_%d' % i)
                .using(default=settings['R_fb_%d' % i], optional=True,
                       validators=[ValueAtLeast(minimum=0)]))
            settings['C_fb_%d' % i] = (self.control_board.calibration.C_fb[i] *
                                       1e12)
            schema_entries.append(
                Float.named('C_fb_%d' % i)
                .using(default=settings['C_fb_%d' % i], optional=True,
                       validators=[ValueAtLeast(minimum=0)]))

        form = Form.of(*schema_entries)

        @gtk_threadsafe
        def _edit_config_dialog():
            dialog = FormViewDialog(form, 'Edit configuration settings')
            valid, response = dialog.run()
            if valid:
                for k, v in response.items():
                    if settings[k] != v:
                        m = re.match('(R|C)_(hv|fb)_(\d)', k)
                        if k == 'amplifier_gain':
                            self.control_board.amplifier_gain = v
                        elif k == 'auto_adjust_amplifier_gain':
                            self.control_board.auto_adjust_amplifier_gain = v
                        elif k == 'WAVEOUT_GAIN_1':
                            self.control_board.waveout_gain_1 = v
                        elif k == 'VGND':
                            self.control_board.vgnd = v
                        elif k == 'SWITCHING_BOARD_I2C_ADDRESS':
                            self.control_board.switching_board_i2c_address = v
                        elif k == 'SIGNAL_GENERATOR_BOARD_I2C_ADDRESS':
                            self.control_board\
                                .signal_generator_board_i2c_address = v
                        elif k == 'voltage_tolerance':
                            self.control_board.voltage_tolerance = v
                        elif k == 'use_antialiasing_filter':
                            self.control_board.use_antialiasing_filter = v
                        elif k == 'max_waveform_voltage':
                            self.control_board.max_waveform_voltage = v
                        elif k == 'min_waveform_frequency':
                            self.control_board.min_waveform_frequency = v
                        elif k == 'max_waveform_frequency':
                            self.control_board.max_waveform_frequency = v
                        elif m:
                            series_resistor = int(m.group(3))
                            if m.group(2) == 'hv':
                                channel = 0
                            else:
                                channel = 1
                            if m.group(1) == 'R':
                                self.control_board.set_series_resistance(
                                    channel, v, resistor_index=series_resistor)
                            else:
                                if v is None:
                                    v = 0
                                self.control_board.set_series_capacitance(
                                    channel, v / 1e12,
                                    resistor_index=series_resistor)
                if get_app().protocol:
                    self.on_step_run()
        _edit_config_dialog()

    def on_reset_configuration_to_default_values(self, widget=None, data=None):
        self.control_board.reset_config_to_defaults()

    def update_connection_status(self):
        '''
        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code, ensuring the
            code runs in the main GTK thread.
        '''
        self.connection_status = "Not connected"
        app = get_app()
        connected = self.control_board.connected()
        if connected:
            name = self.control_board.name()
            version = self.control_board.hardware_version()
            firmware = self.control_board.software_version()
            n_channels = self.control_board.number_of_channels()
            serial_number = self.control_board.serial_number
            self.connection_status = ('%s v%s (Firmware: %s, S/N %03d)\n%d '
                                      'channels' % (name, version, firmware,
                                                    serial_number, n_channels))

        @gtk_threadsafe
        def _update_ui_connected_status():
            # Enable/disable control board menu items based on the connection
            # status of the control board.
            apply_dict_depth_first(self.menu_items, lambda key, node, parents,
                                   *args:
                                   node.item[0].set_sensitive(connected))

            app.main_window_controller.label_control_board_status\
                .set_text(self.connection_status)
        _update_ui_connected_status()

    def on_device_impedance_update(self, results):
        '''
        .. versionchanged:: 2.3.3
            Use :func:`gtk_threadsafe` decorator to wrap GTK code, ensuring the
            code runs in the main GTK thread.
        '''
        app = get_app()

        @gtk_threadsafe
        def _update_ui_impedance():
            label = (self.connection_status + ', Voltage: %.1f V' %
                     results.V_actuation()[-1])

            # add normalized force to the label if we've calibrated the device
            if results.calibration._c_drop:
                label += (u'\nForce: %.1f \u03BCN/mm (c<sub>device</sub>='
                          u'%.1f pF/mm<sup>2</sup>)' %
                          (np.mean(1e6 * results.force(Ly=1.0)), 1e12 *
                           results.calibration.c_drop(results.frequency)))

            app.main_window_controller.label_control_board_status\
                .set_markup(label)
        _update_ui_impedance()

        options = self.get_step_options()

        voltage = results.voltage
        logger.info('[DMFControlBoardPlugin]'
                    '.on_device_impedance_update():')
        logger.info('\tset_voltage=%.1f, measured_voltage=%.1f, '
                    'error=%.1f%%', voltage, results.V_actuation()[-1], 100 *
                    (results.V_actuation()[-1] - voltage) / voltage)

        # check that the signal is within tolerance
        if (abs(results.V_actuation()[-1] - voltage) >
                self.control_board.voltage_tolerance):

            # if the signal is less than the voltage tolerance
            if results.V_actuation()[-1] < self.control_board.voltage_tolerance:
                if self.control_board.auto_adjust_amplifier_gain:
                    # reset the amplifier gain to a high value
                    self.control_board.amplifier_gain = 300
                error_msg = ("Low voltage detected. Please check that the "
                             "amplifier is on.")
                logger.error(error_msg)
                raise ValueError(error_msg, 'low-voltage')

            # allow maximum of 5 adjustment attempts
            if (self.control_board.auto_adjust_amplifier_gain and
                    self.n_voltage_adjustments is not None and
                    self.n_voltage_adjustments < 5):
                logger.info('\tn_voltage_adjustments=%d',
                            self.n_voltage_adjustments)
                emit_signal("set_voltage", voltage,
                            interface=IWaveformGenerator)
                self.check_impedance(options, self.n_voltage_adjustments + 1)
            else:
                self.n_voltage_adjustments = None
                if app.running:
                    self._voltage_tolerance_error_flag = True
                    logger.info('Voltage tolerance exceeded!')
                else:
                    logger.warning('Failed to achieve the specified voltage.')

        if (self.control_board.auto_adjust_amplifier_gain and not
                self.amplifier_gain_initialized):
            self.amplifier_gain_initialized = True
            logger.info('Amplifier gain initialized (gain=%.1f)',
                        self.control_board.amplifier_gain)

    def get_actuated_area(self):
        return self.actuated_area

    def on_step_run(self):
        """
        Handler called whenever a step is executed.

        Plugins that handle this signal must emit the on_step_complete signal
        once they have completed the step. The protocol controller will wait
        until all plugins have completed the current step before proceeding.
        """
        logger.info('[DMFControlBoardPlugin] on_step_run()')
        self._kill_running_step()
        app = get_app()
        options = self.get_step_options()
        feedback_options = options.feedback_options
        app_values = self.get_app_values()

        try:
            if (self.control_board.connected() and (app.realtime_mode or
                                                    app.running)):

                # initialize the amplifier gain
                if (self.control_board.auto_adjust_amplifier_gain and not
                        self.amplifier_gain_initialized):
                    emit_signal("set_frequency",
                                options.frequency,
                                interface=IWaveformGenerator)
                    emit_signal("set_voltage", options.voltage,
                                interface=IWaveformGenerator)
                    self.check_impedance(options)

                max_channels = self.control_board.number_of_channels()
                # All channels should default to off.
                channel_states = np.zeros(max_channels, dtype=int)
                # Set the state of any channels that have been set explicitly.
                channel_states[self.channel_states.index
                               .values.tolist()] = self.channel_states

                if feedback_options.feedback_enabled:
                    if feedback_options.action.__class__ == RetryAction:
                        attempt = app.protocol.current_step_attempt
                        if attempt <= feedback_options.action.max_repeats:
                            frequency = options.frequency
                            if app_values['use_force_normalization'] and \
                                (self.control_board.calibration and
                                 self.control_board.calibration._c_drop):
                                voltage = self.control_board.force_to_voltage(
                                    options.force +
                                    feedback_options.action.increase_force *
                                    attempt,
                                    options.frequency
                                )
                            else:
                                voltage = (options.voltage +
                                           feedback_options.action
                                           .increase_voltage * attempt)
                            emit_signal("set_voltage", voltage,
                                        interface=IWaveformGenerator)
                            if frequency != self.current_frequency:
                                emit_signal("set_frequency", frequency,
                                            interface=IWaveformGenerator)
                                self.check_impedance(options)
                            self.measure_impedance_non_blocking(
                                app_values['sampling_window_ms'],
                                int(math.ceil(options.duration /
                                              (app_values['sampling_window_ms']
                                               + app_values
                                               ['delay_between_windows_ms']))),
                                app_values['delay_between_windows_ms'],
                                app_values['interleave_feedback_samples'],
                                app_values['use_rms'],
                                channel_states)
                            logger.debug('[DMFControlBoardPlugin] on_step_run:'
                                         ' timeout_add(%d, '
                                         '_callback_retry_action_completed)' %
                                         options.duration)
                            self.timeout_id = gobject.timeout_add(
                                options.duration,
                                self._callback_retry_action_completed, options)
                        else:
                            self.step_complete('Fail')
                        return
                    elif (feedback_options.action.__class__ ==
                          SweepFrequencyAction):
                        frequencies = np.logspace(
                            np.log10(feedback_options.action.start_frequency),
                            np.log10(feedback_options.action.end_frequency),
                            int(feedback_options.action.n_frequency_steps)
                        ).tolist()
                        voltage = options.voltage
                        results = FeedbackResultsSeries('Frequency')
                        emit_signal("set_voltage", voltage,
                                    interface=IWaveformGenerator)
                        test_options = deepcopy(options)
                        self._callback_sweep_frequency(test_options,
                                                       results,
                                                       channel_states,
                                                       frequencies,
                                                       first_call=True)
                        return
                    elif (feedback_options.action.__class__ ==
                          SweepVoltageAction):
                        voltages = np.linspace(feedback_options.action
                                               .start_voltage,
                                               feedback_options.action
                                               .end_voltage,
                                               feedback_options.action
                                               .n_voltage_steps).tolist()
                        frequency = options.frequency
                        if frequency != self.current_frequency:
                            emit_signal("set_voltage", options.voltage,
                                        interface=IWaveformGenerator)
                            emit_signal("set_frequency", frequency,
                                        interface=IWaveformGenerator)
                            self.check_impedance(options)
                        results = FeedbackResultsSeries('Voltage')
                        test_options = deepcopy(options)
                        self._callback_sweep_voltage(test_options,
                                                     results,
                                                     channel_states,
                                                     voltages,
                                                     first_call=True)
                        return
                else:
                    emit_signal("set_frequency",
                                options.frequency,
                                interface=IWaveformGenerator)
                    emit_signal("set_voltage", options.voltage,
                                interface=IWaveformGenerator)
                    self.check_impedance(options)
                    self.control_board.state_of_all_channels = channel_states
            # Turn off all electrodes if we're not in realtime mode and not
            # running a protocol.
            elif (self.control_board.connected() and not app.realtime_mode and
                  not app.running):
                logger.info('Turning off all electrodes.')
                self.control_board.set_state_of_all_channels(
                    np.zeros(self.control_board.number_of_channels()))

            # if a protocol is running, wait for the specified minimum duration
            if app.running:
                logger.debug('[DMFControlBoardPlugin] on_step_run: '
                             'timeout_add(%d, _callback_step_completed)' %
                             options.duration)
                self.timeout_id = gobject.timeout_add(
                    options.duration, self._callback_step_completed)
                return
            else:
                self.step_complete()
        except DeviceScaleNotSet:
            logger.error("Please set the area of one of your electrodes.")

    def step_complete(self, return_value=None):
        app = get_app()
        if app.running or app.realtime_mode:
            emit_signal('on_step_complete', [self.name, return_value])

    def on_step_complete(self, plugin_name, return_value=None):
        if plugin_name == self.name:
            self.timeout_id = None

    def get_measure_impedance_data(self):
        """
        This function wraps the control_board.get_measure_impedance_data()
        function and adds the actuated area.
        """
        results = self.control_board.get_measure_impedance_data()
        results.area = self.get_actuated_area()
        return results

    def _kill_running_step(self):
        if self.timeout_id:
            logger.debug('[DMFControlBoardPlugin] _kill_running_step: removing'
                         'timeout_id=%d' % self.timeout_id)
            gobject.source_remove(self.timeout_id)

    def _callback_step_completed(self):
        logger.debug('[DMFControlBoardPlugin] _callback_step_completed')
        self.step_complete()
        return False  # stop the timeout from refiring

    def _callback_retry_action_completed(self, options):
        logger.debug('[DMFControlBoardPlugin] '
                     '_callback_retry_action_completed')
        app = get_app()
        area = self.get_actuated_area()
        return_value = None
        results = self.get_measure_impedance_data()
        logger.debug("V_actuation=%s" % results.V_actuation())
        logger.debug("Z_device=%s" % results.Z_device())
        app.experiment_log.add_data({"FeedbackResults": results}, self.name)

        normalized_capacitance = np.ma.masked_invalid(results.capacitance() /
                                                      area)

        if (self.control_board.calibration._c_drop and
                np.max(normalized_capacitance) <
                options.feedback_options.action.percent_threshold / 100.0 *
                self.control_board.calibration.c_drop(options.frequency)):
            logger.info('step=%d: attempt=%d, max(C)/A=%.1e F/mm^2. Repeat' %
                        (app.protocol.current_step_number,
                         app.protocol.current_step_attempt,
                         np.max(normalized_capacitance)))
            # signal that the step should be repeated
            return_value = 'Repeat'
        else:
            logger.info('step=%d: attempt=%d, max(C)/A=%.1e F/mm^2. OK' %
                        (app.protocol.current_step_number,
                         app.protocol.current_step_attempt,
                         np.max(normalized_capacitance)))
            try:
                emit_signal("on_device_impedance_update", results)
            except ValueError, exception:
                if exception.args[-1] == 'low-voltage':
                    # Low voltage was detected so stop protocol.
                    self.step_complete('Fail')
        self.step_complete(return_value)
        return False  # Stop the timeout from refiring

    def _callback_sweep_frequency(self, options, results, state, frequencies,
                                  first_call=False):
        logger.debug('[DMFControlBoardPlugin] '
                     '_callback_sweep_frequency')
        app = get_app()
        app_values = self.get_app_values()

        # if this isn't the first call, we need to add the data from the
        # previous call
        if not first_call:
            frequency = frequencies.pop(0)
            data = self.get_measure_impedance_data()
            results.add_data(frequency, data)
            logger.debug("V_actuation=%s" % data.V_actuation())
            logger.debug("Z_device=%s" % data.Z_device())

        # if there are frequencies left to sweep
        if len(frequencies):
            frequency = frequencies[0]
            emit_signal("set_frequency", frequency,
                        interface=IWaveformGenerator)
            options.frequency = frequency
            self.measure_impedance_non_blocking(
                app_values['sampling_window_ms'],
                int(math.ceil(options.duration /
                              (app_values['sampling_window_ms'] +
                               app_values['delay_between_windows_ms']))),
                app_values['delay_between_windows_ms'],
                app_values['interleave_feedback_samples'],
                app_values['use_rms'],
                state)
            logger.debug('[DMFControlBoardPlugin] _callback_sweep_frequency: '
                         'timeout_add(%d, _callback_sweep_frequency)' %
                         options.duration)
            self.timeout_id = gobject.timeout_add(options.duration,
                                                  self
                                                  ._callback_sweep_frequency,
                                                  options, results, state,
                                                  frequencies)
        else:
            app.experiment_log.add_data({"FeedbackResultsSeries": results},
                                        self.name)
            self.step_complete()
        return False  # Stop the timeout from refiring

    def _callback_sweep_voltage(self, options, results, state, voltages,
                                first_call=False):
        logger.debug('[DMFControlBoardPlugin] '
                     '_callback_sweep_voltage')
        app = get_app()
        app_values = self.get_app_values()

        # if this isn't the first call, we need to retrieve the data from the
        # previous call
        if not first_call:
            voltage = voltages.pop(0)
            data = self.get_measure_impedance_data()
            results.add_data(voltage, data)
            logger.debug("V_actuation=%s" % data.V_actuation())
            logger.debug("Z_device=%s" % data.Z_device())

        # if there are voltages left to sweep
        if len(voltages):
            voltage = voltages[0]
            emit_signal("set_voltage", voltage,
                        interface=IWaveformGenerator)
            options.voltage = voltage
            self.measure_impedance_non_blocking(
                app_values['sampling_window_ms'],
                int(math.ceil(options.duration /
                              (app_values['sampling_window_ms'] +
                               app_values['delay_between_windows_ms']))),
                app_values['delay_between_windows_ms'],
                app_values['interleave_feedback_samples'],
                app_values['use_rms'],
                state)
            logger.debug('[DMFControlBoardPlugin] _callback_sweep_voltage: '
                         'timeout_add(%d, _callback_sweep_voltage)' %
                         options.duration)
            self.timeout_id = \
                gobject.timeout_add(options.duration,
                                    self._callback_sweep_voltage,
                                    options, results, state, voltages)
        else:
            app.experiment_log.add_data({"FeedbackResultsSeries": results},
                                        self.name)
            self.step_complete()
        return False  # Stop the timeout from refiring

    def on_dmf_device_swapped(self, old_dmf_device, dmf_device):
        self.feedback_options_controller\
            .feedback_options_menu_item.set_sensitive(True)

    def on_protocol_run(self):
        """
        Handler called when a protocol starts running.
        """
        app = get_app()
        self._voltage_tolerance_error_flag = False
        if not self.control_board.connected():
            logger.warning("Warning: no control board connected.")
        elif (self.control_board.number_of_channels() <=
              app.dmf_device.max_channel()):
            logger.warning("Warning: currently connected board does not have "
                           "enough channels for this protocol.")

    def on_protocol_pause(self):
        """
        Handler called when a protocol is paused.
        """
        app = get_app()
        self._kill_running_step()
        if self.control_board.connected() and not app.realtime_mode:
            # Turn off all electrodes
            logger.debug('Turning off all electrodes.')
            self.control_board.set_state_of_all_channels(
                np.zeros(self.control_board.number_of_channels()))
            if self._voltage_tolerance_error_flag:
                logger.warning('Some steps in the protocol failed to achieve '
                               'the specified voltage.')

    def on_experiment_log_selection_changed(self, data):
        """
        Handler called whenever the experiment log selection changes.

        Parameters:
            data : dictionary of experiment log data for the selected steps
        """
        if self.feedback_results_controller:
            self.feedback_results_controller. \
                on_experiment_log_selection_changed(data)

    def set_voltage(self, voltage):
        """
        Set the waveform voltage.

        Parameters:
            voltage : RMS voltage
        """
        logger.info("[DMFControlBoardPlugin].set_voltage(%.1f)" % voltage)
        self.control_board.set_waveform_voltage(voltage)

    def set_frequency(self, frequency):
        """
        Set the waveform frequency.

        Parameters:
            frequency : frequency in Hz
        """
        logger.info("[DMFControlBoardPlugin].set_frequency(%.1f)" % frequency)
        self.control_board.set_waveform_frequency(frequency)
        self.current_frequency = frequency

    def check_impedance(self, options, n_voltage_adjustments=0):
        """
        Check the device impedance.

        Note that this function blocks until it returns.
        """
        # increment the number of adjustment attempts
        self.n_voltage_adjustments = n_voltage_adjustments

        app_values = self.get_app_values()
        test_options = deepcopy(options)
        # take 5 samples to allow signal/gain to stabilize
        test_options.duration = app_values['sampling_window_ms'] * 5
        test_options.feedback_options = FeedbackOptions(feedback_enabled=True,
                                                        action=RetryAction())
        delay_between_windows_ms = 0
        results = \
            self.measure_impedance(
                app_values['sampling_window_ms'],
                int(math.ceil(test_options.duration /
                              (app_values['sampling_window_ms'] +
                               delay_between_windows_ms))),
                delay_between_windows_ms,
                app_values['interleave_feedback_samples'],
                app_values['use_rms'],
                np.zeros(self.control_board.number_of_channels(), dtype=int))
        try:
            emit_signal("on_device_impedance_update", results)
        except ValueError, exception:
            app = get_app()
            if app.running and exception.args[-1] == 'low-voltage':
                # Low voltage was detected so stop protocol.
                self.step_complete('Fail')
        return results

    def _check_n_sampling_windows(self,
                                  sampling_window_ms,
                                  n_sampling_windows,
                                  delay_between_windows_ms):
        # Figure out the maximum number of sampling windows we can collect
        # before filling the serial buffer
        # (MAX_PAYLOAD_LENGTH - 4 * sizeof(float)) /
        #     (NUMBER_OF_ADC_CHANNELS * (sizeof(int8_t) + sizeof(int16_t)))
        n_sampling_windows_max = \
            (self.control_board.MAX_PAYLOAD_LENGTH - 4 * 4) / (2 * (1 + 2))

        # if we're going to exceed this number, adjust the delay between
        # samples
        if n_sampling_windows > n_sampling_windows_max:
            logger.info('[DMFControlBoardPlugin] _check_n_sampling_windows():'
                        ' n_sampling_windows=%d > %d' %
                        (n_sampling_windows, n_sampling_windows_max))
            duration = (sampling_window_ms + delay_between_windows_ms) * \
                n_sampling_windows
            delay_between_windows_ms = math.ceil(float(duration) /
                                                 n_sampling_windows_max -
                                                 sampling_window_ms)
            n_sampling_windows = int(math.floor(duration /
                                                (sampling_window_ms +
                                                 delay_between_windows_ms)))
            logger.info('[DMFControlBoardPlugin] _check_n_sampling_windows():'
                        ' delay_between_windows_ms=%d, n_sampling_windows=%d',
                        delay_between_windows_ms, n_sampling_windows)
            return n_sampling_windows, delay_between_windows_ms
        return n_sampling_windows, delay_between_windows_ms

    def measure_impedance_non_blocking(self,
                                       sampling_window_ms,
                                       n_sampling_windows,
                                       delay_between_windows_ms,
                                       interleave_samples,
                                       rms,
                                       state):
        # wrapper function to adjust the delay between sampling windows if
        # necessary to avoid exceeding the maximum serial buffer length

        n_sampling_windows, delay_between_windows_ms = \
            self._check_n_sampling_windows(sampling_window_ms,
                                           n_sampling_windows,
                                           delay_between_windows_ms)

        self.control_board.measure_impedance_non_blocking(
                                             sampling_window_ms,
                                             n_sampling_windows,
                                             delay_between_windows_ms,
                                             interleave_samples,
                                             rms,
                                             state)

    def measure_impedance(self,
                          sampling_window_ms,
                          n_sampling_windows,
                          delay_between_windows_ms,
                          interleave_samples,
                          rms,
                          state):
        # wrapper function to adjust the delay between sampling windows if
        # necessary to avoid exceeding the maximum serial buffer length

        n_sampling_windows, delay_between_windows_ms = \
            self._check_n_sampling_windows(sampling_window_ms,
                                           n_sampling_windows,
                                           delay_between_windows_ms)

        return self.control_board.measure_impedance(
                                             sampling_window_ms,
                                             n_sampling_windows,
                                             delay_between_windows_ms,
                                             interleave_samples,
                                             rms,
                                             state)

    def get_default_step_options(self):
        return DMFControlBoardOptions()

    def set_step_values(self, values_dict, step_number=None):
        step_number = self.get_step_number(step_number)
        logger.debug('[DMFControlBoardPlugin] set_step[%d]_values(): '
                     'values_dict=%s' % (step_number, values_dict,))
        form = self.StepFields(value=values_dict)
        try:
            if not form.validate():
                errors = ""
                for name, field in form.iteritems():
                    for msg in field.errors:
                        errors += " " + msg
                raise ValueError(errors)
            options = self.get_step_options(step_number=step_number)
            for name, field in form.iteritems():
                if field.value is None:
                    continue
                if name in self._feedback_fields:
                    setattr(options.feedback_options, name, field.value)
                else:
                    setattr(options, name, field.value)
        finally:
            emit_signal('on_step_options_changed', [self.name, step_number],
                        interface=IPlugin)

    def get_step_values(self, step_number=None):
        app = get_app()
        if step_number is None:
            step_number = app.protocol.current_step_number

        options = self.get_step_options(step_number)

        values = {}
        for name in self.StepFields.field_schema_mapping:
            try:
                value = getattr(options, name)
            except AttributeError:
                value = getattr(options.feedback_options, name)
            values[name] = value
        return values

    def get_step_value(self, name, step_number=None):
        app = get_app()
        if name not in self.StepFields.field_schema_mapping:
            raise KeyError('No field with name %s for plugin %s' % (name,
                                                                    self.name))
        if step_number is None:
            step_number = app.protocol.current_step_number
        options = self.get_step_options(step_number)
        try:
            return getattr(options, name)
        except AttributeError:
            return getattr(options.feedback_options, name)

    def on_step_options_changed(self, plugin, step_number):
        app = get_app()
        app_values = self.get_app_values()
        options = self.get_step_options()
        if app_values['use_force_normalization'] and \
            (self.control_board.calibration is not None and
             self.control_board.calibration._c_drop):
            options.voltage = self.control_board.force_to_voltage(
                options.force,
                options.frequency)
        if self.feedback_options_controller:
            (self.feedback_options_controller
             .on_step_options_changed(plugin, step_number))
        if app.protocol and (not app.running and app.realtime_mode and
                             app.protocol.current_step_number == step_number
                             and plugin == self.name):
            self.on_step_run()

    def on_step_swapped(self, original_step_number, new_step_number):
        logger.debug('[DMFControlBoardPlugin] on_step_swapped():'
                     'original_step_number=%d, new_step_number=%d' %
                     (original_step_number, new_step_number))
        self.on_step_options_changed(self.name,
                                     get_app().protocol.current_step_number)

    def on_experiment_log_changed(self, log):
        # Check if the experiment log already has control board meta data, and
        # if so, return.
        data = log.get("control board name")
        for val in data:
            if val:
                return

        # otherwise, add the name, hardware version, serial number,
        # and firmware version
        data = {}
        if self.control_board.connected():
            data["control board name"] = self.control_board.name()
            data["control board serial number"] = \
                self.control_board.serial_number
            data["control board hardware version"] = (self.control_board
                                                      .hardware_version())
            data["control board software version"] = (self.control_board
                                                      .software_version())
            # add info about the devices on the i2c bus
            try:
                data["i2c devices"] = (self.control_board._i2c_devices)
            except Exception:
                pass
        log.add_data(data)

    def on_export_experiment_log_data(self, log):
        feedback_results_df = \
            microdrop_experiment_log_to_feedback_results_df(log)
        step_summary_df =\
            feedback_results_df_to_step_summary_df(feedback_results_df)
        data = {}
        data['feedback results'] = feedback_results_df
        data['step summary'] = step_summary_df
        return data

    def get_schedule_requests(self, function_name):
        """
        Returns a list of scheduling requests (i.e., ScheduleRequest
        instances) for the function specified by function_name.
        """
        if function_name == 'on_step_swapped':
            return [ScheduleRequest('microdrop.electrode_controller_plugin',
                                    self.name)]
        elif function_name in ['on_step_options_changed']:
            return [ScheduleRequest(self.name,
                                    'microdrop.gui.protocol_grid_controller'),
                    ScheduleRequest(self.name,
                                    'microdrop.gui.protocol_controller')]
        elif function_name == 'on_app_options_changed':
            return [ScheduleRequest('microdrop.app', self.name)]
        elif function_name == 'on_protocol_swapped':
            return [ScheduleRequest('microdrop.gui.protocol_grid_controller',
                                    self.name)]
        return []

    def configurations_dir(self):
        directory = path(get_app().config['data_dir']) \
            .joinpath('configurations')
        logger.debug('calibrations_dir=%s', directory)
        if not directory.isdir():
            directory.makedirs_p()
        return directory

    def calibrations_dir(self):
        directory = path(get_app().config['data_dir']).joinpath('calibrations')
        logger.debug('calibrations_dir=%s', directory)
        if not directory.isdir():
            directory.makedirs_p()
        return directory

    def _file_prefix(self):
        timestamp = datetime.now().strftime('%Y-%m-%dT%Hh%Mm%S')
        return '[%05d]-%s-' % (self.control_board.serial_number, timestamp)


PluginGlobals.pop_env()
