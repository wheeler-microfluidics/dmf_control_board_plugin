"""
Copyright 2011 Ryan Fobel

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
from datetime import datetime
from copy import deepcopy
import logging
import math
import os
import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import gtk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.mlab as mlab
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from path_helpers import path
import scipy.optimize as optimize
from matplotlib.backends.backend_gtkagg import (FigureCanvasGTKAgg as
                                                FigureCanvasGTK)
from matplotlib.backends.backend_gtkagg import (NavigationToolbar2GTKAgg as
                                                NavigationToolbar)
from microdrop_utility import SetOfInts, Version, FutureVersionError, is_float
from microdrop_utility.gui import (textentry_validate,
                                   combobox_set_model_from_list,
                                   combobox_get_active_text, text_entry_dialog,
                                   FormViewDialog, yesno)
from flatland.schema import String, Form, Integer, Boolean, Float
from flatland.validation import ValueAtLeast
from microdrop.plugin_manager import (emit_signal, IWaveformGenerator, IPlugin,
                                      get_service_instance_by_name)
from microdrop.app_context import get_app
from microdrop.plugin_helpers import get_plugin_info
from dmf_control_board_firmware import (FeedbackCalibration, FeedbackResults,
                                        FeedbackResultsSeries)
from dmf_control_board_firmware.calibrate.hv_attenuator import (
  plot_feedback_params)
from dmf_control_board_firmware.calibrate.impedance_benchmarks import (
  plot_stat_summary)
from .wizards import (MicrodropImpedanceAssistantView,
                      MicrodropReferenceAssistantView)


class AmplifierGainNotCalibrated(Exception):
    pass


class RetryAction():
    class_version = str(Version(0, 1))

    def __init__(self,
                 percent_threshold=None,
                 increase_voltage=None,
                 max_repeats=None):
        if percent_threshold:
            self.percent_threshold = percent_threshold
        else:
            self.percent_threshold = 0
        if increase_voltage:
            self.increase_voltage = increase_voltage
        else:
            self.increase_voltage = 0
        if max_repeats:
            self.max_repeats = max_repeats
        else:
            self.max_repeats = 3
        self.version = self.class_version

    def __setstate__(self, dict):
        self.__dict__ = dict
        if 'version' not in self.__dict__:
            self.version = str(Version(0, 0))
        self._upgrade()

    def _upgrade(self):
        """
        Upgrade the serialized object if necessary.

        Raises:
            FutureVersionError: file was written by a future version of the
                software.
        """
        logging.debug("[RetryAction]._upgrade()")
        version = Version.fromstring(self.version)
        logging.debug('[RetryAction] version=%s, class_version=%s' %
                      (str(version), self.class_version))
        if version > Version.fromstring(self.class_version):
            logging.debug('[RetryAction] version>class_version')
            raise FutureVersionError(Version.fromstring(self.class_version),
                                     version)
        elif version < Version.fromstring(self.class_version):
            if version < Version(0, 1):
                del self.capacitance_threshold
                self.percent_threshold = 0
                self.version = str(Version(0, 1))
                logging.info('[RetryAction] upgrade to version %s' %
                             self.version)
        else:
            # Else the versions are equal and don't need to be upgraded
            pass


class SweepFrequencyAction():
    def __init__(self,
                 start_frequency=None,
                 end_frequency=None,
                 n_frequency_steps=None):

        service = get_service_instance_by_name(
            get_plugin_info(path(__file__).parent).plugin_name)

        if start_frequency:
            self.start_frequency = start_frequency
        else:
            if service.control_board.connected():
                self.start_frequency = \
                    service.control_board.min_waveform_frequency
            else:
                self.start_frequency = 100
        if end_frequency:
            self.end_frequency = end_frequency
        else:
            if service.control_board.connected():
                self.end_frequency = \
                    service.control_board.max_waveform_frequency
            else:
                self.end_frequency = 20e3
        if n_frequency_steps:
            self.n_frequency_steps = n_frequency_steps
        else:
            self.n_frequency_steps = 10


class SweepVoltageAction():
    def __init__(self,
                 start_voltage=None,
                 end_voltage=None,
                 n_voltage_steps=None):
        if start_voltage:
            self.start_voltage = start_voltage
        else:
            self.start_voltage = 5
        if end_voltage:
            self.end_voltage = end_voltage
        else:
            self.end_voltage = 100
        if n_voltage_steps:
            self.n_voltage_steps = n_voltage_steps
        else:
            self.n_voltage_steps = 20


class SweepElectrodesAction():
    def __init__(self,
                 channels=None):
        if channels:
            self.channels = channels
        else:
            self.channels = SetOfInts()
            app = get_app()
            for e in app.dmf_device.electrodes.values():
                self.channels.update(e.channels)


class FeedbackOptions():
    """
    This class stores the feedback options for a single step in the protocol.
    """
    class_version = str(Version(0, 1))

    def __init__(self, feedback_enabled=None,
                 action=None):
        if feedback_enabled:
            self.feedback_enabled = feedback_enabled
        else:
            self.feedback_enabled = True
        if action:
            self.action = action
        else:
            self.action = RetryAction()
        self.version = self.class_version

    def _upgrade(self):
        """
        Upgrade the serialized object if necessary.

        Raises:
            FutureVersionError: file was written by a future version of the
                software.
        """
        logging.debug('[FeedbackOptions]._upgrade()')
        if hasattr(self, 'version'):
            version = Version.fromstring(self.version)
        else:
            version = Version(0)
        logging.debug('[FeedbackOptions] version=%s, class_version=%s' %
                      (str(version), self.class_version))
        if version > Version.fromstring(self.class_version):
            logging.debug('[FeedbackOptions] version>class_version')
            raise FutureVersionError(Version.fromstring(self.class_version),
                                     version)
        elif version < Version.fromstring(self.class_version):
            if version < Version(0, 1):
                del self.sampling_time_ms
                del self.n_samples
                del self.delay_between_samples_ms
            self.version = self.class_version
        # else the versions are equal and don't need to be upgraded


class FeedbackOptionsController():
    def __init__(self, plugin):
        self.plugin = plugin
        self.builder = gtk.Builder()
        self.builder.add_from_file(path(__file__).parent
                                   .joinpath('glade',
                                             'feedback_options.glade'))
        self.window = self.builder.get_object("window")
        self.builder.connect_signals(self)
        self.window.set_title("Feedback Options")
        self.initialized = False

    def on_plugin_enable(self):
        if not self.initialized:
            app = get_app()
            self.feedback_options_menu_item = gtk.MenuItem("Feedback Options")
            self.plugin.control_board_menu.append(self
                                                  .feedback_options_menu_item)
            self.feedback_options_menu_item.connect("activate",
                                                    self.on_window_show)
            self.feedback_options_menu_item.show()
            self.feedback_options_menu_item.set_sensitive(
                app.dmf_device is not None)

            self.measure_cap_filler_menu_item = gtk.MenuItem("Measure "
                                                             "capacitance of "
                                                             "filler media")
            app.dmf_device_controller.view.popup.add_item(
                self.measure_cap_filler_menu_item)
            self.measure_cap_filler_menu_item.connect("activate",
                                                      self
                                                      .on_measure_cap_filler)
            self.measure_cap_liquid_menu_item = gtk.MenuItem("Measure "
                                                             "capacitance of "
                                                             "liquid")
            app.dmf_device_controller.view.popup.add_item(
                self.measure_cap_liquid_menu_item)
            self.measure_cap_liquid_menu_item.connect("activate",
                                                      self
                                                      .on_measure_cap_liquid)

            self.initialized = True
        self.measure_cap_filler_menu_item.show()
        self.measure_cap_liquid_menu_item.show()

    def on_plugin_disable(self):
        self.measure_cap_filler_menu_item.hide()
        self.measure_cap_liquid_menu_item.hide()

    def on_window_show(self, widget, data=None):
        """
        Handler called when the user clicks on "Feedback Options" in the
        "Tools" menu.
        """
        options = self.plugin.get_step_options().feedback_options
        self._set_gui_sensitive(options)
        self._update_gui_state(options)
        self.window.show()

    def on_window_delete_event(self, widget, data=None):
        """
        Handler called when the user closes the "Feedback Options" window.
        """
        self.window.hide()
        return True

    def on_measure_cap_filler(self, widget, data=None):
        self.plugin.control_board.calibration._C_filler = \
            self.measure_device_capacitance()

    def on_measure_cap_liquid(self, widget, data=None):
        self.plugin.control_board.calibration._C_drop = \
            self.measure_device_capacitance()
            
    def measure_device_capacitance(self):
        app = get_app()
        area = self.plugin.get_actuated_area()

        if area == 0:
            logging.error("At least one electrode must be actuated to perform "
                          "calibration.")
            return

        step = app.protocol.current_step()
        dmf_options = step.get_data(self.plugin.name)

        # get the current state of channels
        state = app.dmf_device_controller.get_step_options().state_of_channels

        voltage = dmf_options.voltage
        emit_signal("set_voltage", voltage, interface=IWaveformGenerator)
        app_values = self.plugin.get_app_values()
        test_options = deepcopy(dmf_options)
        test_options.duration = 5 * app_values['sampling_window_ms']
        test_options.feedback_options = FeedbackOptions(
            feedback_enabled=True, action=SweepFrequencyAction()
        )

        results = FeedbackResultsSeries('Frequency')
        for frequency in np.logspace(
            np.log10(test_options.feedback_options.action.start_frequency),
            np.log10(test_options.feedback_options.action.end_frequency),
            int(test_options.feedback_options.action.n_frequency_steps)
        ):
            emit_signal("set_frequency", frequency,
                        interface=IWaveformGenerator)
            data = self.plugin.measure_impedance(
                app_values['sampling_window_ms'],
                int(math.ceil(test_options.duration /
                              (app_values['sampling_window_ms'] + \
                               app_values['delay_between_windows_ms']))),
                app_values['delay_between_windows_ms'],
                app_values['interleave_feedback_samples'],
                app_values['use_rms'],
                state)
            results.add_data(frequency, data)
            results.area = area
            capacitance = np.mean(results.capacitance())
            logging.info('\tcapacitance = %e F (%e F/mm^2)' % \
                     (capacitance, capacitance / area))
            
        capacitance = np.mean(results.capacitance())
        logging.info('mean(capacitance) = %e F (%e F/mm^2)' % \
                     (capacitance, capacitance / area))

        # set the frequency back to it's original state
        emit_signal("set_frequency",
                    dmf_options.frequency,
                    interface=IWaveformGenerator)
        self.plugin.check_impedance(dmf_options)
        
        # turn off all electrodes if we're not in realtime mode
        if not app.realtime_mode:
            self.plugin.control_board.set_state_of_all_channels(
                np.zeros(self.plugin.control_board.number_of_channels())
            )
            
        return dict(frequency=results.frequency.tolist(),
                    capacitance=(np.mean(results.capacitance(), 1) / area). \
                        tolist())

    def on_button_feedback_enabled_toggled(self, widget, data=None):
        """
        Handler called when the "Feedback enabled" check box is toggled.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.feedback_enabled = widget.get_active()
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_step_options_changed(self, plugin_name, step_number):
        app = get_app()
        if (self.plugin.name == plugin_name and
                app.protocol.current_step_number == step_number):
            all_options = self.plugin.get_step_options(step_number)
            options = all_options.feedback_options
            self._set_gui_sensitive(options)
            self._update_gui_state(options)

    def _update_gui_state(self, options):
        # update the state of the "Feedback enabled" check button
        button = self.builder.get_object("button_feedback_enabled")
        if options.feedback_enabled != button.get_active():
            # Temporarily disable radio-button toggled signal handler to avoid
            # infinite loop (handler emits signal that results in this method
            # being called).
            button.handler_block_by_func(self
                                         .on_button_feedback_enabled_toggled)
            button.set_active(options.feedback_enabled)
            button.handler_unblock_by_func(self
                                           .on_button_feedback_enabled_toggled)

        # update the retry action parameters
        retry = (options.action.__class__ == RetryAction)
        if retry:
            self.builder.get_object("textentry_percent_threshold")\
                .set_text(str(options.action.percent_threshold))
            self.builder.get_object("textentry_increase_voltage")\
                .set_text(str(options.action.increase_voltage))
            self.builder.get_object("textentry_max_repeats").set_text(
                str(options.action.max_repeats))
        else:
            self.builder.get_object("textentry_percent_threshold")\
                .set_text("")
            self.builder.get_object("textentry_increase_voltage").set_text("")
            self.builder.get_object("textentry_max_repeats").set_text("")
        button = self.builder.get_object("radiobutton_retry")
        if retry != button.get_active():
            # Temporarily disable toggled signal handler (see above)
            button.handler_block_by_func(self.on_radiobutton_retry_toggled)
            button.set_active(retry)
            button.handler_unblock_by_func(self.on_radiobutton_retry_toggled)

        sweep_frequency = (options.action.__class__ == SweepFrequencyAction)
        # update the sweep frequency action parameters
        if sweep_frequency:
            self.builder.get_object("textentry_start_frequency")\
                .set_text(str(options.action.start_frequency / 1000.0))
            self.builder.get_object("textentry_end_frequency").set_text(
                str(options.action.end_frequency / 1000.0))
            self.builder.get_object("textentry_n_frequency_steps").set_text(
                str(str(options.action.n_frequency_steps)))
        else:
            self.builder.get_object("textentry_start_frequency").set_text("")
            self.builder.get_object("textentry_end_frequency").set_text("")
            self.builder.get_object("textentry_n_frequency_steps").set_text("")
        button = self.builder.get_object("radiobutton_sweep_frequency")
        if sweep_frequency != button.get_active():
            # Temporarily disable toggled signal handler (see above)
            button.handler_block_by_func(
                self.on_radiobutton_sweep_frequency_toggled)
            button.set_active(sweep_frequency)
            button.handler_unblock_by_func(
                self.on_radiobutton_sweep_frequency_toggled)

        sweep_voltage = (options.action.__class__ == SweepVoltageAction)
        # update the sweep voltage action parameters
        if sweep_voltage:
            self.builder.get_object("textentry_start_voltage")\
                .set_text(str(options.action.start_voltage))
            self.builder.get_object("textentry_end_voltage").set_text(
                str(options.action.end_voltage))
            self.builder.get_object("textentry_n_voltage_steps").set_text(
                str(str(options.action.n_voltage_steps)))
        else:
            self.builder.get_object("textentry_start_voltage").set_text("")
            self.builder.get_object("textentry_end_voltage").set_text("")
            self.builder.get_object("textentry_n_voltage_steps").set_text("")
        button = self.builder.get_object("radiobutton_sweep_voltage")
        if sweep_voltage != button.get_active():
            # Temporarily disable toggled signal handler (see above)
            button.handler_block_by_func(self
                                         .on_radiobutton_sweep_voltage_toggled)
            button.set_active(sweep_voltage)
            button.handler_unblock_by_func(
                self.on_radiobutton_sweep_voltage_toggled)

        sweep_electrodes = (options.action.__class__ == SweepElectrodesAction)
        # update the sweep electrodes action parameters
        if sweep_electrodes:
            self.builder.get_object("textentry_channels")\
                .set_text(str(options.action.channels))
        else:
            self.builder.get_object("textentry_channels").set_text("")
        button = self.builder.get_object("radiobutton_sweep_electrodes")
        if sweep_electrodes != button.get_active():
            # Temporarily disable toggled signal handler (see above)
            button.handler_block_by_func(
                self.on_radiobutton_sweep_electrodes_toggled)
            button.set_active(sweep_electrodes)
            button.handler_unblock_by_func(
                self.on_radiobutton_sweep_electrodes_toggled)

    def _set_gui_sensitive(self, options):
        self.builder.get_object("radiobutton_retry")\
            .set_sensitive(options.feedback_enabled)
        self.builder.get_object("radiobutton_sweep_frequency")\
            .set_sensitive(options.feedback_enabled)
        self.builder.get_object("radiobutton_sweep_voltage")\
            .set_sensitive(options.feedback_enabled)
        self.builder.get_object("radiobutton_sweep_electrodes")\
            .set_sensitive(options.feedback_enabled)

        retry = (options.action.__class__ == RetryAction)
        self.builder.get_object("textentry_percent_threshold")\
            .set_sensitive(options.feedback_enabled and retry)
        self.builder.get_object("textentry_increase_voltage")\
            .set_sensitive(options.feedback_enabled and retry)
        self.builder.get_object("textentry_max_repeats")\
            .set_sensitive(options.feedback_enabled and retry)

        sweep_frequency = (options.action.__class__ == SweepFrequencyAction)
        self.builder.get_object("textentry_start_frequency")\
            .set_sensitive(options.feedback_enabled and sweep_frequency)
        self.builder.get_object("textentry_end_frequency")\
            .set_sensitive(options.feedback_enabled and sweep_frequency)
        self.builder.get_object("textentry_n_frequency_steps")\
            .set_sensitive(options.feedback_enabled and sweep_frequency)

        sweep_voltage = (options.action.__class__ == SweepVoltageAction)
        self.builder.get_object("textentry_start_voltage")\
            .set_sensitive(options.feedback_enabled and sweep_voltage)
        self.builder.get_object("textentry_end_voltage")\
            .set_sensitive(options.feedback_enabled and sweep_voltage)
        self.builder.get_object("textentry_n_voltage_steps")\
            .set_sensitive(options.feedback_enabled and sweep_voltage)

        sweep_electrodes = (options.action.__class__ == SweepElectrodesAction)
        self.builder.get_object("textentry_channels")\
            .set_sensitive(options.feedback_enabled and sweep_electrodes)

    def on_radiobutton_retry_toggled(self, widget, data=None):
        """
        Handler called when the "Retry until capacitance..." radio button is
        toggled.
        """
        logging.debug('retry was toggled %s' % (('OFF',
                                                 'ON')[widget.get_active()]))
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        retry = widget.get_active()
        if retry and options.action.__class__ != RetryAction:
            options.action = RetryAction()
        if retry:
            emit_signal('on_step_options_changed',
                        [self.plugin.name, app.protocol.current_step_number],
                        interface=IPlugin)

    def on_radiobutton_sweep_frequency_toggled(self, widget, data=None):
        """
        Handler called when the "Sweep Frequency..." radio button is
        toggled.
        """
        logging.debug('sweep_frequency was toggled %s' %
                      (('OFF', 'ON')[widget.get_active()]))
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        sweep_frequency = widget.get_active()
        if (sweep_frequency and options.action.__class__ !=
                SweepFrequencyAction):
            options.action = SweepFrequencyAction()
        if sweep_frequency:
            emit_signal('on_step_options_changed',
                        [self.plugin.name, app.protocol.current_step_number],
                        interface=IPlugin)

    def on_radiobutton_sweep_voltage_toggled(self, widget, data=None):
        """
        Handler called when the "Sweep Voltage..." radio button is
        toggled.
        """
        logging.debug('sweep_voltage was toggled %s' %
                      (('OFF', 'ON')[widget.get_active()]))
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        sweep_voltage = widget.get_active()
        if sweep_voltage and options.action.__class__ != SweepVoltageAction:
            options.action = SweepVoltageAction()
        if sweep_voltage:
            emit_signal('on_step_options_changed',
                        [self.plugin.name, app.protocol.current_step_number],
                        interface=IPlugin)

    def on_radiobutton_sweep_electrodes_toggled(self, widget, data=None):
        """
        Handler called when the "Sweep Electrodes..." radio button is
        toggled.
        """
        logging.debug('sweep_electrodes was toggled %s' %
                      (('OFF', 'ON')[widget.get_active()]))
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        sweep_electrodes = widget.get_active()
        if (sweep_electrodes and options.action.__class__ !=
                SweepElectrodesAction):
            options.action = SweepElectrodesAction()
        if sweep_electrodes:
            emit_signal('on_step_options_changed',
                        [self.plugin.name, app.protocol.current_step_number],
                        interface=IPlugin)

    def on_textentry_percent_threshold_focus_out_event(self,
                                                       widget,
                                                       event):
        """
        Handler called when the "percent threshold" text box loses focus.
        """
        self.on_textentry_percent_threshold_changed(widget)
        return False

    def on_textentry_percent_threshold_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "percent
        threshold" text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_percent_threshold_changed(widget)

    def on_textentry_percent_threshold_changed(self, widget):
        """
        Update the percent threshold value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.percent_threshold = textentry_validate(
            widget, options.action.percent_threshold, float)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_increase_voltage_focus_out_event(self, widget, event):
        """
        Handler called when the "increase voltage" text box loses focus.
        """
        self.on_textentry_increase_voltage_changed(widget)
        return False

    def on_textentry_increase_voltage_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "increase
        voltage" text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_increase_voltage_changed(widget)

    def on_textentry_increase_voltage_changed(self, widget):
        """
        Update the increase voltage value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.increase_voltage = textentry_validate(
            widget, options.action.increase_voltage, float)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_max_repeats_focus_out_event(self, widget, event):
        """
        Handler called when the "max repeats" text box loses focus.
        """
        self.on_textentry_max_repeats_changed(widget)
        return False

    def on_textentry_max_repeats_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "max repeats"
        text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_max_repeats_changed(widget)

    def on_textentry_max_repeats_changed(self, widget):
        """
        Update the max repeats value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.max_repeats = textentry_validate(
            widget, options.action.max_repeats, int)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_start_frequency_focus_out_event(self, widget, event):
        """
        Handler called when the "start frequency" text box loses focus.
        """
        self.on_textentry_start_frequency_changed(widget)
        return False

    def on_textentry_start_frequency_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "start frequency"
        text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_start_frequency_changed(widget)

    def on_textentry_start_frequency_changed(self, widget):
        """
        Update the start frequency value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.start_frequency = textentry_validate(
            widget, options.action.start_frequency / 1e3, float) * 1e3
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_end_frequency_focus_out_event(self, widget, event):
        """
        Handler called when the "end frequency" text box loses focus.
        """
        self.on_textentry_end_frequency_changed(widget)
        return False

    def on_textentry_end_frequency_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "end frequency"
        text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_end_frequency_changed(widget)

    def on_textentry_end_frequency_changed(self, widget):
        """
        Update the end frequency value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.end_frequency = textentry_validate(
            widget, options.action.end_frequency / 1e3, float) * 1e3
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_n_frequency_steps_focus_out_event(self, widget, event):
        """
        Handler called when the "number of frequency steps" text box loses
        focus.
        """
        self.on_textentry_n_frequency_steps_changed(widget)
        return False

    def on_textentry_n_frequency_steps_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "number of
        frequency steps" text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_n_frequency_steps_changed(widget)

    def on_textentry_n_frequency_steps_changed(self, widget):
        """
        Update the number of frequency steps value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.n_frequency_steps = textentry_validate(
            widget, options.action.n_frequency_steps, float)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_start_voltage_focus_out_event(self, widget, event):
        """
        Handler called when the "start voltage" text box loses focus.
        """
        self.on_textentry_start_voltage_changed(widget)
        return False

    def on_textentry_start_voltage_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "start voltage"
        text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_start_voltage_changed(widget)

    def on_textentry_start_voltage_changed(self, widget):
        """
        Update the start voltage value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.start_voltage = textentry_validate(
            widget, options.action.start_voltage, float)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_end_voltage_focus_out_event(self, widget, event):
        """
        Handler called when the "end voltage" text box loses focus.
        """
        self.on_textentry_end_voltage_changed(widget)
        return False

    def on_textentry_end_voltage_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "end voltage"
        text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_end_voltage_changed(widget)

    def on_textentry_end_voltage_changed(self, widget):
        """
        Update the end voltage value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.end_voltage = textentry_validate(
            widget, options.action.end_voltage, float)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_n_voltage_steps_focus_out_event(self, widget, event):
        """
        Handler called when the "number of voltage steps" text box loses focus.
        """
        self.on_textentry_n_voltage_steps_changed(widget)
        return False

    def on_textentry_n_voltage_steps_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "number of
        voltage steps" text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_n_voltage_steps_changed(widget)

    def on_textentry_n_voltage_steps_changed(self, widget):
        """
        Update the number of voltage steps value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        options.action.n_voltage_steps = textentry_validate(
            widget, options.action.n_voltage_steps, float)
        emit_signal('on_step_options_changed',
                    [self.plugin.name, app.protocol.current_step_number],
                    interface=IPlugin)

    def on_textentry_channels_focus_out_event(self, widget, event):
        """
        Handler called when the "electrodes" text box loses focus.
        """
        self.on_textentry_channels_changed(widget)
        return False

    def on_textentry_channels_key_press_event(self, widget, event):
        """
        Handler called when the user presses a key within the "electrodes"
        text box.
        """
        if event.keyval == 65293:  # user pressed enter
            self.on_textentry_channels_changed(widget)

    def on_textentry_channels_changed(self, widget):
        """
        Update the electrodes value for the current step.
        """
        app = get_app()
        all_options = self.plugin.get_step_options()
        options = all_options.feedback_options
        try:
            channels = SetOfInts(widget.get_text())
            assert(min(channels) >= 0)
            options.action.channels = channels
            emit_signal('on_step_options_changed',
                        [self.plugin.name, app.protocol.current_step_number],
                        interface=IPlugin)
        except:
            widget.set_text(str(options.action.channels))


class FeedbackResultsController():
    def __init__(self, plugin):
        self.plugin = plugin
        self.builder = gtk.Builder()
        app = get_app()
        self.builder.add_from_file(path(__file__).parent
                                   .joinpath('glade',
                                             'feedback_results.glade'))
        self.window = self.builder.get_object("window")
        self.combobox_x_axis = self.builder.get_object("combobox_x_axis")
        self.combobox_y_axis = self.builder.get_object("combobox_y_axis")
        self.checkbutton_normalize_by_area = self.builder.get_object(
            "checkbutton_normalize_by_area")
        self.checkbutton_filter = self.builder.get_object(
            "checkbutton_filter")
        self.window.set_title("Feedback Results")
        self.builder.connect_signals(self)
        self.data = []

        self.feedback_results_menu_item = gtk.MenuItem("Feedback Results")
        app.main_window_controller.menu_view.append(
            self.feedback_results_menu_item)
        self.feedback_results_menu_item.connect("activate",
                                                self.on_window_show)

        self.figure = Figure()
        self.canvas = FigureCanvasGTK(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.vbox = self.builder.get_object("vbox1")
        toolbar = NavigationToolbar(self.canvas, self.window)
        self.vbox.pack_start(self.canvas)
        self.vbox.pack_start(toolbar, False, False)
        combobox_set_model_from_list(self.combobox_x_axis,
                                     ["Time", "Frequency", "Voltage"])
        combobox_set_model_from_list(self.combobox_y_axis,
                                     ["Impedance", "Capacitance", "Velocity",
                                      "Voltage", "x-position"])
        self.combobox_x_axis.set_active(0)
        self.combobox_y_axis.set_active(0)

    def on_window_show(self, widget, data=None):
        """
        Handler called when the user clicks on "Feedback Results" in the "View"
        menu.
        """
        self.window.show_all()

    def on_window_delete_event(self, widget, data=None):
        """
        Handler called when the user closes the "Feedback Results" window.
        """
        self.window.hide()
        return True

    def on_combobox_x_axis_changed(self, widget, data=None):
        x_axis = combobox_get_active_text(self.combobox_x_axis)
        if x_axis == "Time":
            combobox_set_model_from_list(self.combobox_y_axis, ["Impedance",
                                                                "Capacitance",
                                                                "Velocity",
                                                                "Voltage",
                                                                "x-position"])
        else:
            combobox_set_model_from_list(self.combobox_y_axis, ["Impedance",
                                                                "Capacitance",
                                                                "Voltage"])
        self.combobox_y_axis.set_active(0)
        self.update_plot()

    def on_combobox_y_axis_changed(self, widget, data=None):
        y_axis = combobox_get_active_text(self.combobox_y_axis)
        self.checkbutton_normalize_by_area.set_sensitive(y_axis == "Impedance"
                                                         or y_axis ==
                                                         "Capacitance")
        self.checkbutton_filter.set_sensitive(y_axis == "Impedance" or \
                                              y_axis == "Capacitance" or \
                                              y_axis == "Velocity" or \
                                              y_axis == "x-position")
        self.update_plot()

    def on_checkbutton_normalize_by_area_toggled(self, widget, data=None):
        self.update_plot()

    def on_checkbutton_filter_toggled(self, widget, data=None):
        self.update_plot()

    def on_export_data_clicked(self, widget, data=None):
        dialog = gtk.FileChooserDialog(title="Export data",
                                       action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                       buttons=(gtk.STOCK_CANCEL,
                                                gtk.RESPONSE_CANCEL,
                                                gtk.STOCK_SAVE,
                                                gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_current_name("export.csv")
        filter = gtk.FileFilter()
        filter.set_name("*.csv")
        filter.add_pattern("*.csv")
        dialog.add_filter(filter)
        filter = gtk.FileFilter()
        filter.set_name("All files")
        filter.add_pattern("*")
        dialog.add_filter(filter)
        response = dialog.run()
        if response == gtk.RESPONSE_OK:
            filename = dialog.get_filename()
            logging.info("Exporting to file %s." % filename)
            try:
                with open(filename, 'w') as f:
                    f.write("\n".join(self.export_data))
            except Exception, e:
                logging.error("Problem exporting file. %s." % e)
        dialog.destroy()

    def on_experiment_log_selection_changed(self, data):
        """
        Handler called whenever the experiment log selection changes.

        :param data: experiment log data (list of dictionaries, one per step)
                     for the selected steps
        """
        self.data = data
        self.update_plot()

    def update_plot(self):
        x_axis = combobox_get_active_text(self.combobox_x_axis)
        y_axis = combobox_get_active_text(self.combobox_y_axis)
        self.axis.cla()
        self.axis.grid(True)
        legend = []
        legend_loc = "upper right"
        self.export_data = []
        experiment_log_controller = get_service_instance_by_name(
            "microdrop.gui.experiment_log_controller", "microdrop")
        protocol = experiment_log_controller.results.protocol
        dmf_device = experiment_log_controller.results.dmf_device

        normalization_string = ""
        if self.checkbutton_normalize_by_area.get_active():
            normalization_string = "/mm$^2$"

        if y_axis == "Impedance":
            self.axis.set_title("Impedance%s" % normalization_string)
            self.axis.set_ylabel("|Z$_{device}$| ($\Omega$%s)" %
                                 normalization_string)
            self.axis.set_yscale('log')
        elif y_axis == "Capacitance":
            self.axis.set_title("Capacitance%s" % normalization_string)
            self.axis.set_ylabel("C$_{device}$ (F%s)" % normalization_string)
            legend_loc = "lower right"
        elif y_axis == "Velocity":
            self.axis.set_title("Instantaneous velocity")
            self.axis.set_ylabel("Velocity$_{drop}$ (mm/s)")
        elif y_axis == "Voltage":
            self.axis.set_title("Actuation voltage")
            self.axis.set_ylabel("V$_{actuation}$ (V$_{RMS}$)")
            legend_loc = "lower right"
        elif y_axis == "x-position":
            self.axis.set_title("x-position")
            self.axis.set_ylabel("x-position (mm)")

        handles = []
        if x_axis == "Time":
            self.axis.set_xlabel("Time (ms)")
            for row in self.data:
                if (self.plugin.name in row.keys() and "FeedbackResults" in
                        row[self.plugin.name].keys()):
                    results = row[self.plugin.name]["FeedbackResults"]
                    state_of_channels = protocol[row['core']["step"]]. \
                        get_data('microdrop.gui.dmf_device_controller'). \
                        state_of_channels
                    area = dmf_device.actuated_area(state_of_channels)
                    results.area = area

                    normalization = 1.0
                    if self.checkbutton_normalize_by_area.get_active():
                        if area == 0:
                            continue
                        else:
                            normalization = area

                    self.export_data.append('step:, %d' % (row['core']["step"]
                                                           + 1))
                    self.export_data.append('step time (s):, %f' %
                                            (row['core']["time"]))

                    # only plot values that have a valid fb and hv resistor,
                    # and that have been using the same fb and hv resistor for
                    # > 1 consecutive measurement
                    ind = mlab.find(np.logical_and(
                        np.logical_and(results.fb_resistor != -1,
                                       results.hv_resistor != -1),
                        np.logical_and(
                            np.concatenate(([0], np.diff(results.fb_resistor)))
                            == 0,
                            np.concatenate(([0], np.diff(results.hv_resistor)))
                            == 0)))

                    if y_axis == "Impedance":
                        Z = results.Z_device()
                        if self.checkbutton_filter.get_active():
                            lines = self.axis.plot(results.time,
                                results.Z_device(filter_order=3) / \
                                normalization)
                            c = matplotlib.colors.colorConverter.to_rgba(
                                lines[0].get_c(), alpha=.2)
                            handles.append(lines[0])
                            self.axis.plot(results.time, Z / normalization,
                                           color=c)
                        else:
                            self.axis.plot(results.time, Z / normalization)
                        self.export_data.append('time (ms):, ' +
                                                ", ".join([str(x) for x in
                                                           results.time]))
                        self.export_data.append('impedance (Ohms%s):, ' %
                                                (normalization_string) +
                                                ", ".join([str(x) for x in
                                                           Z / normalization]))
                    elif y_axis == "Capacitance":
                        C = results.capacitance()
                        if self.checkbutton_filter.get_active():
                            C_filtered = results.capacitance(filter_order=2)
                            lines = self.axis.plot(results.time,
                                C_filtered / normalization)
                            handles.append(lines[0])
                            c = matplotlib.colors.colorConverter.to_rgba(
                                lines[0].get_c(), alpha=.2)
                            self.axis.plot(results.time, C / normalization,
                                           color=c)
                        else:
                            self.axis.plot(results.time, C / normalization)
                        self.export_data.append('time (ms):, ' +
                                                ", ".join([str(x) for x in
                                                           results.time]))
                        self.export_data.append('capacitance (F%s):,' %
                                                normalization_string +
                                                ", ".join([str(x) for x in
                                                           C / normalization]))
                    elif y_axis == "Velocity":
                        if self.checkbutton_filter.get_active():
                            t, dxdt = results.dxdt(filter_order=3)
                            lines = self.axis.plot(t, dxdt * 1000)
                            handles.append(lines[0])
                            c = matplotlib.colors.colorConverter.to_rgba(
                                lines[0].get_c(), alpha=.2)
                            t, dxdt = results.dxdt()
                            self.axis.plot(t, dxdt * 1000, color=c)
                        else:
                            t, dxdt = results.dxdt()
                            self.axis.plot(t, dxdt * 1000)
                        self.export_data.append('time (ms):, ' +
                                                ", ".join([str(x) for x in t]))
                        self.export_data.append('velocity (mm/s):,' +
                                                ", ".join([str(x) for x in
                                                           dxdt]))
                    elif y_axis == "Voltage":
                        self.axis.plot(results.time[ind],
                                       results.V_actuation()[ind])
                        self.export_data.append('time (ms):, ' +
                                                ", ".join([str(x) for x in
                                                           results.time[ind]]))
                        self.export_data.append('V_actuation (V_RMS):,' +
                                                ", ".join([str(x) for x in
                                                           results
                                                           .V_actuation()
                                                           [ind]]))
                    elif y_axis == "x-position":
                        x_pos = results.x_position()
                        if self.checkbutton_filter.get_active():
                            lines = self.axis.plot(results.time,
                                results.x_position(filter_order=3))
                            handles.append(lines[0])
                            c = matplotlib.colors.colorConverter.to_rgba(
                                lines[0].get_c(), alpha=.2)
                            self.axis.plot(results.time, x_pos, color=c)
                        else:
                            self.axis.plot(results.time, x_pos)
                        self.export_data.append('time (ms):, ' +
                                                ", ".join([str(x) for x in
                                                           results.time]))
                        self.export_data.append('velocity (mm/s):,' +
                                                ", ".join([str(x) for x in
                                                           x_pos]))
                    legend.append("Step %d (%.3f s)" % (row['core']["step"] +
                                                        1,
                                                        row['core']["time"]))
        elif x_axis == "Frequency":
            self.axis.set_xlabel("Frequency (Hz)")
            self.axis.set_xscale('log')
            for row in self.data:
                if (self.plugin.name in row.keys() and "FeedbackResultsSeries"
                        in row[self.plugin.name].keys()):
                    results = row[self.plugin.name]["FeedbackResultsSeries"]

                    if results.xlabel != "Frequency":
                        continue

                    state_of_channels = protocol[row['core']["step"]]. \
                        get_data('microdrop.gui.dmf_device_controller'). \
                        state_of_channels
                    area = dmf_device.actuated_area(state_of_channels)

                    normalization = 1.0
                    if self.checkbutton_normalize_by_area.get_active():
                        if area == 0:
                            continue
                        else:
                            normalization = area

                    self.export_data.append('step:, %d' %
                                            (row['core']["step"] + 1))
                    self.export_data.append('step time (s):, %f' %
                                            (row['core']["time"]))
                    self.export_data.append('frequency (Hz):, ' +
                                            ", ".join([str(x) for x in
                                                       results.frequency]))
                    if y_axis == "Impedance":
                        Z = np.ma.masked_invalid(results.Z_device())
                        self.axis.errorbar(results.frequency,
                                           np.mean(Z, 1) /
                                           normalization,
                                           np.std(Z, 1) /
                                           normalization, fmt='.')
                        self.export_data.append('mean(impedance) (Ohms%s):, ' %
                                                normalization_string +
                                                ", ".join([str(x) for x in
                                                           np.mean(Z, 1) /
                                                           normalization]))
                        self.export_data.append('std(impedance) (Ohms%s):, ' %
                                                normalization_string +
                                                ", ".join([str(x) for x in
                                                           np.std(Z, 1) /
                                                           normalization]))
                    elif y_axis == "Capacitance":
                        C = np.ma.masked_invalid(results.capacitance())
                        self.axis.errorbar(results.frequency,
                                           np.mean(C, 1) /
                                           normalization,
                                           np.std(C, 1) /
                                           normalization, fmt='.')
                        self.export_data.append('mean(capacitance) (F):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.mean(C /
                                                               normalization,
                                                               1)]))
                        self.export_data.append('std(capacitance/area) (F):, '
                                                + ", "
                                                .join([str(x) for x in
                                                       np.std(C, 1) /
                                                       normalization]))
                    elif y_axis == "Voltage":
                        V = np.ma.masked_invalid(results.V_actuation())
                        self.axis.errorbar(results.frequency,
                                           np.mean(V, 1),
                                           np.std(V, 1),
                                           fmt='.')
                        self.export_data.append('mean(V_actuation) (Vrms):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.mean(V, 1)]))
                        self.export_data.append('std(V_actuation) (Vrms):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.std(V, 1)]))
                    legend.append("Step %d (%.3f s)" % (row['core']["step"] +
                                                        1,
                                                        row['core']["time"]))
        elif x_axis == "Voltage":
            self.axis.set_xlabel("Actuation Voltage (V$_{RMS}$)")
            for row in self.data:
                if (self.plugin.name in row.keys() and "FeedbackResultsSeries"
                        in row[self.plugin.name].keys()):
                    results = row[self.plugin.name]["FeedbackResultsSeries"]

                    if results.xlabel != "Voltage":
                        continue

                    state_of_channels = protocol[row['core']["step"]]. \
                        get_data('microdrop.gui.dmf_device_controller'). \
                        state_of_channels
                    area = dmf_device.actuated_area(state_of_channels)

                    normalization = 1.0
                    if self.checkbutton_normalize_by_area.get_active():
                        if area == 0:
                            continue
                        else:
                            normalization = area

                    self.export_data.append('step:, %d' %
                                            (row['core']["step"] + 1))
                    self.export_data.append('step time (s):, %f' %
                                            (row['core']["time"]))
                    self.export_data.append('voltage (Vrms):, ' +
                                            ", ".join([str(x) for x in
                                                       results.voltage]))
                    if y_axis == "Impedance":
                        Z = np.ma.masked_invalid(results.Z_device())
                        self.axis.errorbar(results.voltage,
                                           np.mean(Z, 1) /
                                           normalization,
                                           np.std(Z, 1) /
                                           normalization, fmt='.')
                        self.export_data.append('mean(impedance) (Ohms%s):, ' %
                                                normalization_string +
                                                ", ".join([str(x) for x in
                                                           np.mean(Z, 1) /
                                                           normalization]))
                        self.export_data.append('std(impedance) (Ohms%s):, '
                                                % normalization_string +
                                                ", ".join([str(x) for x in
                                                           np.std(Z, 1) /
                                                           normalization]))
                    elif y_axis == "Capacitance":
                        C = np.ma.masked_invalid(results.capacitance())
                        self.axis.errorbar(results.voltage,
                                           np.mean(C, 1) /
                                           normalization,
                                           np.std(C, 1) /
                                           normalization, fmt='.')
                        self.export_data.append('mean(capacitance) (F):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.mean(C, 1) /
                                                       normalization]))
                        self.export_data.append('std(capacitance) (F):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.std(C, 1) /
                                                       normalization]))
                    elif y_axis == "Voltage":
                        V = np.ma.masked_invalid(results.V_actuation())
                        self.axis.errorbar(results.voltage,
                                           np.mean(V, 1),
                                           np.std(V, 1),
                                           fmt='.')
                        self.export_data.append('mean(V_actuation) (Vrms):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.mean(V, 1)]))
                        self.export_data.append('std(V_actuation) (Vrms):, ' +
                                                ", "
                                                .join([str(x) for x in
                                                       np.std(V, 1)]))
                    legend.append("Step %d (%.3f s)" % (row['core']["step"] +
                                                        1,
                                                        row['core']["time"]))
        if len(legend):
            if len(handles):
                self.axis.legend(handles, legend, loc=legend_loc)
            else:
                self.axis.legend(legend, loc=legend_loc)

        self.figure.subplots_adjust(left=0.17, bottom=0.15)
        self.canvas.draw()


class FeedbackCalibrationController():
    def __init__(self, plugin):
        self.plugin = plugin
        self.experiment_log_controller = get_service_instance_by_name(
            "microdrop.gui.experiment_log_controller", "microdrop")

    def on_save_log_calibration(self, widget, data=None):
        selected_data = self.experiment_log_controller.get_selected_data()
        calibration = None
        if len(selected_data) > 1:
            logging.error("Multiple steps are selected. Please choose a "
                          "single step.")
            return
        try:
            if 'FeedbackResults' in selected_data[0][self.plugin.name]:
                calibration = (selected_data[0][self.plugin.name]
                               ['FeedbackResults'].calibration)
            elif 'FeedbackResultsSeries' in selected_data[0][self.plugin.name]:
                calibration = (selected_data[0][self.plugin.name]
                               ['FeedbackResultsSeries'].calibration)
        except:
            logging.error("This step does not contain any calibration data.")
            return

        dialog = gtk.FileChooserDialog(title="Save feedback calibration",
                                       action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                       buttons=(gtk.STOCK_CANCEL,
                                                gtk.RESPONSE_CANCEL,
                                                gtk.STOCK_SAVE,
                                                gtk.RESPONSE_OK))

        while True:
            try:
                dialog.set_default_response(gtk.RESPONSE_OK)
                response = dialog.run()
                if response == gtk.RESPONSE_OK:
                    filename = path(dialog.get_filename())
                    with open(filename.abspath(), 'wb') as f:
                        pickle.dump(calibration, f)
                    break
                else:
                    break
            except Exception, why:
                logging.error("Error saving calibration file. %s." % why)
        dialog.destroy()

    def on_load_log_calibration(self, widget, data=None):
        dialog = gtk.FileChooserDialog(
            title="Load calibration from file",
            action=gtk.FILE_CHOOSER_ACTION_OPEN,
            buttons=(gtk.STOCK_CANCEL,
                     gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN,
                     gtk.RESPONSE_OK)
        )
        dialog.set_default_response(gtk.RESPONSE_OK)
        response = dialog.run()
        calibration = None
        if response == gtk.RESPONSE_OK:
            filename = path(dialog.get_filename())
            with open(filename, 'rb') as f:
                try:
                    calibration = pickle.load(f)
                    logging.debug("Loaded object from pickle.")
                    if str(calibration.__class__).split('.')[-1] != \
                            'FeedbackCalibration':
                        raise ValueError()
                except Exception, why:
                    logging.error('Not a valid calibration file.')
                    logging.debug(why)
        dialog.destroy()

        selected_data = self.experiment_log_controller.get_selected_data()
        for row in selected_data:
            try:
                if 'FeedbackResults' in row[self.plugin.name]:
                    row[self.plugin.name]['FeedbackResults'].calibration = \
                        deepcopy(calibration)
                elif 'FeedbackResultsSeries' in row[self.plugin.name]:
                    row[self.plugin.name]['FeedbackResultsSeries'].\
                        calibration = deepcopy(calibration)
            except:
                continue
        # save the experiment log with the new values
        filename = os.path.join(self.experiment_log_controller.results.
                                log.directory,
                                str(self.experiment_log_controller.results.
                                    log.experiment_id),
                                'data')
        self.experiment_log_controller.results.log.save(filename)
        emit_signal("on_experiment_log_selection_changed", [selected_data])

    def on_edit_log_calibration(self, widget, data=None):
        logging.debug("on_edit_log_calibration()")
        settings = {}
        schema_entries = []
        calibration_list = []
        selected_data = self.experiment_log_controller.get_selected_data()

        # Create a list containing the following calibration sections:
        #
        #    * `FeedbackResults`
        #    * `FeedbackResultsSeries`
        for row in selected_data:
            data = None
            if self.plugin.name not in row.keys():
                continue

            if 'FeedbackResults' in row[self.plugin.name]:
                data = row[self.plugin.name]['FeedbackResults']
            elif 'FeedbackResultsSeries' in row[self.plugin.name]:
                data = row[self.plugin.name]['FeedbackResultsSeries']
            else:
                # The current row does not contain any results.
                continue

            calibration_list.append(data.calibration)
            frequency = data.frequency

            # There is a calibration result entry in this row, and it has been
            # added to the end of `calibration_list`.  Therefore, to retrieve
            # it, we retrieve the last item in `calibration_list`.
            calibration = calibration_list[-1]

            # Set default for each setting only if all selected steps have the
            # same value.  Otherwise, leave the default blank.
            if len(calibration_list) == 1:
                # If we only have one calibration result entry, set the default
                # value for the edit dialog to the value from the result entry.
                settings["C_drop"] = calibration.C_drop(frequency)
                settings["C_filler"] = calibration.C_filler(frequency)
                for i in range(len(calibration.R_hv)):
                    settings['R_hv_%d' % i] = calibration.R_hv[i]
                    settings['C_hv_%d' % i] = calibration.C_hv[i]
                for i in range(len(calibration.R_fb)):
                    settings['R_fb_%d' % i] = calibration.R_fb[i]
                    settings['C_fb_%d' % i] = calibration.C_fb[i]
            else:
                # More than one calibration result is selected. If a value has
                # already been set for a resistor or capacitor value and the
                # corresponding value from the current calibration result entry
                # is different, set the default editor value to `None`.
                def check_group_value(name, new):
                    if settings[name] and settings[name] != new:
                        settings[name] = None
                check_group_value("C_drop", calibration.C_drop(frequency))
                check_group_value("C_filler", calibration.C_filler(frequency))
                for i in range(len(calibration.R_hv)):
                    check_group_value('R_hv_%d' % i, calibration.R_hv[i])
                    check_group_value('C_hv_%d' % i, calibration.C_hv[i])
                for i in range(len(calibration.R_fb)):
                    check_group_value('R_fb_%d' % i, calibration.R_fb[i])
                    check_group_value('C_fb_%d' % i, calibration.C_fb[i])

        # no calibration objects?
        if not calibration_list:
            return

        def set_field_value(name, multiplier=1):
            string_value = ""
            if name in settings.keys() and settings[name]:
                string_value = str(settings[name] * multiplier)
            schema_entries.append(String.named(name).using(
                default=string_value, optional=True))

        set_field_value('C_drop', 1e12)
        set_field_value('C_filler', 1e12)

        for i in range(len(calibration.R_hv)):
            set_field_value('R_hv_%d' % i)
            set_field_value('C_hv_%d' % i, 1e12)
        for i in range(len(calibration.R_fb)):
            set_field_value('R_fb_%d' % i)
            set_field_value('C_fb_%d' % i, 1e12)

        form = Form.of(*sorted(schema_entries, key=lambda x: x.name))
        dialog = FormViewDialog('Edit calibration settings')
        valid, response = dialog.run(form)

        if not valid:
            return

        logging.debug("Applying updated calibration settings to log file.")

        def get_field_value(name, multiplier=1):
            try:
                logging.debug('response[%s]=' % name, response[name])
                logging.debug('settings[%s]=' % name, settings[name])
                if (response[name] and (settings[name] is None or
                                        abs(float(response[name]) / multiplier
                                            - settings[name]) / settings[name]
                                        > .0001)):
                    return float(response[name]) / multiplier
            except ValueError:
                logging.error('%s value (%s) is invalid.' %
                              (name, response[name]))
            return None

        value = get_field_value('C_drop', 1e12)
        if value:
            for calibration in calibration_list:
                calibration._C_drop = value
        value = get_field_value('C_filler', 1e12)
        if value:
            for calibration in calibration_list:
                calibration._C_filler = value
        for i in range(len(calibration.R_hv)):
            value = get_field_value('R_hv_%d' % i)
            if value:
                for calibration in calibration_list:
                    calibration.R_hv[i] = value
            value = get_field_value('C_hv_%d' % i, 1e12)
            if value:
                for calibration in calibration_list:
                    calibration.C_hv[i] = value
        for i in range(len(calibration.R_fb)):
            value = get_field_value('R_fb_%d' % i)
            if value:
                for calibration in calibration_list:
                    calibration.R_fb[i] = value
            value = get_field_value('C_fb_%d' % i, 1e12)
            if value:
                for calibration in calibration_list:
                    calibration.C_fb[i] = value

        # save the experiment log with the new values
        filename = os.path.join(self.experiment_log_controller.results
                                .log.directory,
                                str(self.experiment_log_controller.results
                                    .log.experiment_id), 'data')
        self.experiment_log_controller.results.log.save(filename)
        emit_signal("on_experiment_log_selection_changed", [selected_data])

    def on_perform_calibration(self, widget, data=None):
        if not self.plugin.control_board.connected():
            logging.error("A control board must be connected in order to "
                          "perform calibration.")
            return

        self.calibrate_attenuators()

    def load_reference_calibration(self):
        dialog = gtk.FileChooserDialog(
            title="Load reference calibration readings from file",
            action=gtk.FILE_CHOOSER_ACTION_OPEN,
            buttons=(gtk.STOCK_CANCEL,
                     gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN,
                     gtk.RESPONSE_OK)
        )
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_current_folder(self.plugin.calibrations_dir())
        response = dialog.run()
        filename = dialog.get_filename()
        dialog.destroy()

        if response != gtk.RESPONSE_OK:
            return

        try:
            hv_readings = pd.read_hdf(str(filename),
                                        '/feedback/reference/measurements')
            fitted_params = pd.read_hdf(str(filename),
                                        '/feedback/reference/fitted_params')

            figure = Figure(figsize=(14, 8), dpi=60)
            axis = figure.add_subplot(111)

            plot_feedback_params(self.plugin.control_board.calibration
                                 .hw_version.major, hv_readings, fitted_params,
                                 axis=axis)

            window = gtk.Window()
            window.set_default_size(800, 600)
            window.set_title('Fitted reference feedback parameters')
            canvas = FigureCanvasGTK(figure)
            toolbar = NavigationToolbar(canvas, window)
            vbox = gtk.VBox()
            vbox.pack_start(canvas)
            vbox.pack_start(toolbar, False, False)
            window.add(vbox)
            window.show_all()

            logging.info(str(fitted_params))
        except KeyError:
            logging.error('Error loading reference calibration data.')
            return

    def load_impedance_calibration(self):
        dialog = gtk.FileChooserDialog(
            title="Load device load calibration readings from file",
            action=gtk.FILE_CHOOSER_ACTION_OPEN,
            buttons=(gtk.STOCK_CANCEL,
                     gtk.RESPONSE_CANCEL,
                     gtk.STOCK_OPEN,
                     gtk.RESPONSE_OK)
        )
        dialog.set_default_response(gtk.RESPONSE_OK)
        dialog.set_current_folder(self.plugin.calibrations_dir())
        response = dialog.run()
        filename = dialog.get_filename()
        dialog.destroy()

        if response != gtk.RESPONSE_OK:
            return

        try:
            measurements = pd.read_hdf(str(filename),
                                       '/feedback/impedance/measurements')

            figure = Figure(figsize=(14, 14), dpi=60)

            plot_stat_summary(measurements, fig=figure)

            window = gtk.Window()
            window.set_default_size(800, 600)
            window.set_title('Impedance feedback measurement accuracy')
            canvas = FigureCanvasGTK(figure)
            toolbar = NavigationToolbar(canvas, window)
            vbox = gtk.VBox()
            vbox.pack_start(canvas)
            vbox.pack_start(toolbar, False, False)
            window.add(vbox)
            window.show_all()
        except KeyError:
            logging.error('Error loading device load calibration data.')
            return

    def calibrate_attenuators(self):
        calibrations_dir = self.plugin.calibrations_dir()
        configurations_dir = self.plugin.configurations_dir()
        prefix = self.plugin._file_prefix()
        view = MicrodropReferenceAssistantView(self.plugin.control_board)

        def on_calibrated(assistant):
            self.plugin.to_yaml(configurations_dir.joinpath(prefix +
                                                            'config.yml')
            )
            view.to_hdf(
                calibrations_dir.joinpath(prefix + 
                                          'calibration-reference-load.h5')
            )

        # Save the persistent configuration settings from the control-board to
        # a file upon successful calibration.
        view.widget.connect('close', on_calibrated)
        view.show()

    def calibrate_impedance(self):
        calibrations_dir = self.plugin.calibrations_dir()
        configurations_dir = self.plugin.configurations_dir()
        prefix = self.plugin._file_prefix()
        view = MicrodropImpedanceAssistantView(self.plugin.control_board)

        def on_calibrated(assistant):
            self.plugin.to_yaml(configurations_dir.joinpath(prefix +
                                                            'config.yml')
            )
            view.to_hdf(calibrations_dir.joinpath(prefix +
                                                  'calibration-device-load.h5')
            )

        # Save the persistent configuration settings from the control-board to
        # a file upon successful calibration.
        view.widget.connect('close', on_calibrated)
        view.show()
