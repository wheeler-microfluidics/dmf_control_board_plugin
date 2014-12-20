from dmf_control_board_firmware.gui.reference import (AssistantView as
                                             ReferenceAssistantView)
from dmf_control_board_firmware.gui.impedance import (AssistantView as
                                             ImpedanceAssistantView)
from dmf_control_board_firmware.gui.channels import (AssistantView as
                                            ChannelsAssistantView)


class MicrodropReferenceAssistantView(ReferenceAssistantView):
    def create_ui(self):
        super(MicrodropReferenceAssistantView, self).create_ui()
        self.widget.set_modal(True)

    def close_button_clicked(self, assistant):
        self.widget.set_modal(False)
        assistant.hide()

    def cancel_button_clicked(self, assistant):
        self.widget.set_modal(False)
        assistant.hide()


class MicrodropImpedanceAssistantView(ImpedanceAssistantView):
    def create_ui(self):
        super(MicrodropImpedanceAssistantView, self).create_ui()
        self.widget.set_modal(True)

    def close_button_clicked(self, assistant):
        self.widget.set_modal(False)
        assistant.hide()

    def cancel_button_clicked(self, assistant):
        self.widget.set_modal(False)
        assistant.hide()


class MicrodropChannelsAssistantView(ChannelsAssistantView):
    def create_ui(self):
        super(MicrodropChannelsAssistantView, self).create_ui()
        self.widget.set_modal(True)

    def close_button_clicked(self, assistant):
        self.widget.set_modal(False)
        assistant.hide()

    def cancel_button_clicked(self, assistant):
        self.widget.set_modal(False)
        assistant.hide()
