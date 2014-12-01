#!/usr/bin/env python
import numpy as np
import gtk
from dmf_control_board import DMFControlBoard
from pygtkhelpers.delegates import WindowView, SlaveView
from pygtkhelpers.ui.form_view_dialog import create_form_view
from flatland.schema import String, Form, Integer, Boolean, Float
from flatland.validation import ValueAtLeast
from IPython.display import display
from dmf_control_board.calibrate.hv_attenuator import (
    resistor_max_actuation_readings, fit_feedback_params,
    update_control_board_calibration, plot_feedback_params)
from dmf_control_board.calibrate.oscope import (VISA_AVAILABLE, AgilentOscope,
                                                read_oscope as read_oscope_)


if VISA_AVAILABLE:
    oscope = AgilentOscope()
    read_oscope = lambda: oscope.read_ac_vrms()
else:
    read_oscope = read_oscope_


class AssistantView(WindowView):
    def __init__(self, control_board):
        super(AssistantView, self).__init__(self)
        self.control_board = control_board

    def create_ui(self):
        self.widget = gtk.Assistant()
        self.widget.set_default_size(800, 600)
        self.widget.connect("prepare", self.assitant_prepared)
        self.widget.connect("cancel", self.cancel_button_clicked)
        self.widget.connect("close", self.close_button_clicked)
        self.widget.connect("apply", self.apply_button_clicked)

        box = gtk.VBox()
        self.widget.append_page(box)
        self.widget.set_page_type(box, gtk.ASSISTANT_PAGE_INTRO)
        self.widget.set_page_title(box, "Page 1: Introduction")
        content = ('This wizard will guide you through the process of '
                   'calibrating the high-voltage reference load feedback '
                   'measurement circuit.  This feedback circuit is used to '
                   'measure the output voltage of the amplifier on the control'
                   'board.\n\nSee '
                   r'<a href="http://microfluidics.utoronto.ca/trac/dropbot/wiki/Control board calibration#high-voltage-attenuation-calibration">'
                   'here</a> for more details.')
        label = gtk.Label(content)
        label.set_use_markup(True)
        label.set_line_wrap(True)
        box.pack_start(label, True, True, 0)
        self.widget.set_page_complete(box, True)

        box = gtk.VBox()
        self.widget.append_page(box)
        self.widget.set_page_type(box, gtk.ASSISTANT_PAGE_CONTENT)
        self.widget.set_page_title(box, "Page 2: Connect hardware")
        label = gtk.Label(' - Connect DropBot "<tt>Out to Amp</tt>" to amplifier input.\n'
                          ' - Use T-splitter to connect amplifier output to:\n'
                          '   1) DropBot "<tt>In from Amp</tt>".\n'
                          '   2) Oscilloscope input.')
        label.set_line_wrap(True)
        label.set_use_markup(True)
        box.pack_start(label, True, True, 0)
        self.widget.set_page_complete(box, True)

        minimum = 100
        maximum = 20e3
        form = Form.of(
            Integer.named('start_frequency').using(
                default=minimum, optional=True,
                validators=[ValueAtLeast(minimum=minimum), ]),
            Integer.named('end_frequency').using(
                default=maximum, optional=True,
                validators=[ValueAtLeast(minimum=minimum), ]),
            Integer.named('number_of_steps').using(
                default=10, optional=True,
                validators=[ValueAtLeast(minimum=1), ]),
        )
        self.form_view = create_form_view(form)
        self.form_view.form.proxies.connect('changed', display)
        self.widget.append_page(self.form_view.widget)
        self.widget.set_page_type(self.form_view.widget, gtk.ASSISTANT_PAGE_CONTENT)
        self.widget.set_page_title(self.form_view.widget, "Page 2a: Select calibration frequencies")
        self.widget.set_page_complete(self.form_view.widget, True)

        box1 = gtk.VBox()
        self.widget.append_page(box1)
        self.widget.set_page_type(box1, gtk.ASSISTANT_PAGE_PROGRESS)
        self.widget.set_page_title(box1, "Page 3: Record measurements")
        self.measurements_label = gtk.Label('Measurements taken: 0 / ?')
        self.measurements_label.set_line_wrap(True)
        box1.pack_start(self.measurements_label, True, True, 0)
        checkbutton = gtk.CheckButton("Mark page as complete")
        checkbutton.connect("toggled", self.checkbutton_toggled)
        box1.pack_start(checkbutton, False, False, 0)
        self.box1 = box1

        box = gtk.VBox()
        self.widget.append_page(box)
        self.widget.set_page_type(box, gtk.ASSISTANT_PAGE_CONFIRM)
        self.widget.set_page_title(box, "Page 4: Confirm")
        label = gtk.Label("The 'Confirm' page may be set as the final page in the Assistant, however this depends on what the Assistant does. This page provides an 'Apply' button to explicitly set the changes, or a 'Go Back' button to correct any mistakes.")
        label.set_line_wrap(True)
        box.pack_start(label, True, True, 0)
        self.widget.set_page_complete(box, True)

        box = gtk.VBox()
        self.widget.append_page(box)
        self.widget.set_page_type(box, gtk.ASSISTANT_PAGE_SUMMARY)
        self.widget.set_page_title(box, "Page 5: Summary")
        label = gtk.Label("A 'Summary' should be set as the final page of the Assistant if used however this depends on the purpose of your self.widget. It provides information on the changes that have been made during the configuration or details of what the user should do next. On this page only a Close button is displayed. Once at the Summary page, the user cannot return to any other page.")
        label.set_line_wrap(True)
        box.pack_start(label, True, True, 0)
        self.widget.set_page_complete(box, True)

    def assitant_prepared(self, assistant, *args):
        print 'Page %s prepared.' % assistant.get_current_page()
        if assistant.get_current_page() < 3:
            self.widget.set_page_complete(self.box1, False)
        elif assistant.get_current_page() == 3:
            settings = dict([(f, self.form_view.form.fields[f].proxy
                              .get_widget_value())
                             for f in ('start_frequency', 'number_of_steps',
                                       'end_frequency')])
            start_frequency = np.array(settings['start_frequency'])
            end_frequency = np.array(settings['end_frequency'])
            number_of_steps = np.array(settings['number_of_steps'])
            frequencies = np.logspace(np.log10(start_frequency),
                                      np.log10(end_frequency), number_of_steps)
            gtk.idle_add(self.read_measurements, frequencies)

    def read_measurements(self, frequencies):
        try:
            self.measurement_count = len(frequencies) * 4 + 1
            self.measurement_i = 0
            self.measurements_label.set_label('Measurements taken: 0 / %d' %
                                                self.measurement_count)
            self.hv_readings = resistor_max_actuation_readings(
                self.control_board, frequencies, self.read_oscope)
            self.widget.set_page_complete(self.box1, True)
        except StopIteration:
            self.measurements_label.set_label('Measurements taken: 0 / %d' %
                                              self.measurement_count)
            self.widget.set_current_page(2)

    def read_oscope(self):
        result = read_oscope()
        self.measurement_i += 1
        self.measurements_label.set_label('Measurements taken: %d / %d' %
                                          (self.measurement_i,
                                           self.measurement_count))
        return result

    def apply_button_clicked(self, assistant):
        print("The 'Apply' button has been clicked")

    def close_button_clicked(self, assistant):
        print("The 'Close' button has been clicked")
        gtk.main_quit()

    def cancel_button_clicked(self, assistant):
        print("The 'Cancel' button has been clicked")
        gtk.main_quit()

    def checkbutton_toggled(self, checkbutton):
        self.widget.set_page_complete(self.box1, checkbutton.get_active())


if __name__ == '__main__':
    control_board = DMFControlBoard()
    control_board.connect()
    view = AssistantView(control_board)
    view.show_and_run()
