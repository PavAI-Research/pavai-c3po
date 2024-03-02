import gradio as gr
import pandas as pd
import os
import time
import pavai.translator.filedata as filedata
import pavai.translator.translator as translator
import pavai.translator.lang_list as lang_list
import pavai.vocei_web.translator_ui

class CommunicationTranslator:
    def build_translator_ui(self):
        self.blocks_translator = gr.Blocks(analytics_enabled=False)
        with self.blocks_translator:
            state = gr.State()
            with gr.Row():
                with gr.Column(scale=7):
                    ## Seamless Communication ##
                    with gr.Accordion("Speech-to-Speech Translator", open=True):
                        gr.Markdown("Speak on your own native language to another party using realtime speech-to-speech translation.")
                        with gr.Group(visible=True) as voicechat:
                            with gr.Row():
                                with gr.Column(scale=5):
                                    source_language = gr.Dropdown(
                                        label="User Spoken language",
                                        choices=lang_list.ASR_TARGET_LANGUAGE_NAMES,
                                        value="English",
                                    )
                                    input_audio = gr.Audio(
                                        sources=["microphone", "upload"],
                                        label="User voice",
                                        type="filepath",
                                        scale=1,
                                        min_width=20,
                                    )
                                    # user_speech_text = gr.Text(label="Your voice text", lines=2)
                                    btn_translate = gr.Button(
                                        "Translate User Speech to Party",
                                        size="sm",
                                        scale=1,
                                    )
                                with gr.Column(scale=5):
                                    party_source_language = gr.Dropdown(
                                        label="Party Spoken language",
                                        choices=lang_list.ASR_TARGET_LANGUAGE_NAMES,
                                        value="French",
                                    )
                                    party_input_audio = gr.Audio(
                                        sources=["microphone", "upload"],
                                        label="Party voice",
                                        type="filepath",
                                        scale=1,
                                        min_width=20,
                                    )
                                    # party_speech_text = gr.Text(label="Party voice text", lines=2)
                                    party_btn_translate = gr.Button(
                                        "Translate Party Speech to User",
                                        size="sm",
                                        scale=1,
                                    )

                            with gr.Row():
                                with gr.Column():
                                    with gr.Group():
                                        target_language = gr.Dropdown(
                                            label="Target Party language",
                                            choices=lang_list.S2ST_TARGET_LANGUAGE_NAMES,
                                            value=translator.DEFAULT_TARGET_LANGUAGE,
                                        )
                                        xspeaker = gr.Slider(
                                            1,
                                            100,
                                            value=7,
                                            step=1,
                                            label="Speaker Id",
                                            interactive=True,
                                        )
                                        output_audio = gr.Audio(
                                            label="Translated speech",
                                            autoplay=True,
                                            streaming=False,
                                            type="numpy",
                                        )
                                        output_text = gr.Textbox(
                                            label="Translated text", lines=3
                                        )
                                        btn_clear = gr.ClearButton(
                                            [
                                                source_language,
                                                input_audio,
                                                output_audio,
                                                output_text,
                                            ],
                                            size="sm",
                                        )

                                ## Other Party
                                with gr.Column():
                                    with gr.Group():
                                        party_target_language = gr.Dropdown(
                                            label="Target User language",
                                            choices=lang_list.S2ST_TARGET_LANGUAGE_NAMES,
                                            value="English",
                                        )
                                        party_xspeaker = gr.Slider(
                                            1,
                                            100,
                                            value=7,
                                            step=1,
                                            label="Speaker Id",
                                            interactive=True,
                                        )
                                        party_output_audio = gr.Audio(
                                            label="Translated speech",
                                            autoplay=True,
                                            streaming=False,
                                            type="numpy",
                                        )
                                        party_output_text = gr.Textbox(
                                            label="Translated text", lines=3
                                        )
                                        party_btn_clear = gr.ClearButton(
                                            [
                                                party_source_language,
                                                party_input_audio,
                                                party_output_audio,
                                                party_output_text,
                                            ],
                                            size="sm",
                                        )

                                        # handle speaker id change
                                        party_xspeaker.change(
                                            fn=translator.update_value, inputs=xspeaker
                                        )
                                # handle
                                btn_translate.click(
                                    fn=translator.run_s2st,
                                    inputs=[
                                        input_audio,
                                        source_language,
                                        target_language,
                                        xspeaker,
                                    ],
                                    outputs=[output_audio, output_text],
                                    api_name="s2st",
                                )

                                # handle
                                party_btn_translate.click(
                                    fn=translator.run_s2st,
                                    inputs=[
                                        party_input_audio,
                                        party_source_language,
                                        party_target_language,
                                        party_xspeaker,
                                    ],
                                    outputs=[party_output_audio, party_output_text],
                                    api_name="s2st_party",
                                )

                                ## auto submit
                                input_audio.stop_recording(
                                    fn=translator.run_s2st,
                                    inputs=[
                                        input_audio,
                                        source_language,
                                        target_language,
                                    ],
                                    outputs=[output_audio, output_text],
                                )

                                ## auto submit
                                party_input_audio.stop_recording(
                                    fn=translator.run_s2st,
                                    inputs=[
                                        party_input_audio,
                                        party_source_language,
                                        party_target_language,
                                    ],
                                    outputs=[party_output_audio, party_output_text],
                                )
        return self.blocks_translator

# class ScratchPad:
#     def build_scratchpad_ui(self):
#         self.blocks_scratchpad = gr.Blocks(analytics_enabled=False)
#         with self.blocks_scratchpad:
#             with gr.Accordion("Important tasks to complete", open=True):
#                 box_notepad = gr.TextArea(
#                     lines=3,
#                     info="write single page tasks.",
#                     interactive=True,
#                 )
#                 with gr.Row():
#                     with gr.Column(scale=3):
#                         btn_save_notepad = gr.Button(
#                             size="sm", value="save and update"
#                         )
#                     with gr.Column(scale=1):
#                         btn_load_notepad = gr.Button(size="sm", value="load")

#                 ## function: scratch pad
#                 def save_notespad(text_notes):
#                     gr.Info("saved notes to tasks.txt!")
#                     return filedata.save_text_file("tasks.txt", text_notes)

#                 def load_notespad(filename: str = "tasks.txt"):
#                     gr.Info("load notes")
#                     return filedata.get_text_file(filename)

#                 btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
#                 btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
#                 ## update expert mode and speech style
#             with gr.Accordion("Notes", open=False):
#                 box_notepad = gr.TextArea(
#                     lines=5,
#                     info="write single page quick notes.",
#                     interactive=True,
#                 )
#                 with gr.Row():
#                     with gr.Column(scale=3):
#                         btn_save_notepad = gr.Button(
#                             size="sm", value="save and update"
#                         )
#                     with gr.Column(scale=1):
#                         btn_load_notepad = gr.Button(size="sm", value="load")

#                 ## function: scratch pad
#                 def save_notespad(text_notes):
#                     gr.Info("saved notes to notes.txt!")
#                     return filedata.save_text_file("notes.txt", text_notes)

#                 def load_notespad(filename: str = "notes.txt"):
#                     gr.Info("load notes")
#                     return filedata.get_text_file(filename)

#                 btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
#                 btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
#                 ## update expert mode and speech style
#             with gr.Accordion("Todos", open=False):
#                 box_notepad = gr.TextArea(
#                     lines=3,
#                     info="write single page of Todos.",
#                     interactive=True,
#                 )
#                 with gr.Row():
#                     with gr.Column(scale=3):
#                         btn_save_notepad = gr.Button(
#                             size="sm", value="save and update"
#                         )
#                     with gr.Column(scale=1):
#                         btn_load_notepad = gr.Button(size="sm", value="load")

#                 ## function: scratch pad
#                 def save_notespad(text_notes):
#                     gr.Info("saved notes to todos.txt!")
#                     return filedata.save_text_file("todos.txt", text_notes)

#                 def load_notespad(filename: str = "todos.txt"):
#                     gr.Info("load notes")
#                     return filedata.get_text_file(filename)

#                 btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
#                 btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
#                 ## update expert mode and speech style
#             with gr.Accordion("Reminders", open=False):
#                 box_notepad = gr.TextArea(
#                     lines=3,
#                     info="write single page of Todos.",
#                     interactive=True,
#                 )
#                 with gr.Row():
#                     with gr.Column(scale=3):
#                         btn_save_notepad = gr.Button(
#                             size="sm", value="save and update"
#                         )
#                     with gr.Column(scale=1):
#                         btn_load_notepad = gr.Button(size="sm", value="load")

#                 ## function: scratch pad
#                 def save_notespad(text_notes):
#                     gr.Info("saved notes to reminder.txt!")
#                     return filedata.save_text_file("reminder.txt", text_notes)

#                 def load_notespad(filename: str = "reminder.txt"):
#                     gr.Info("load notes")
#                     return filedata.get_text_file(filename)

#                 btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
#                 btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
#                 ## update expert mode and speech style
#         return self.blocks_scratchpad
    
# # theme=gr.themes.Monochrome()
# demoui = gr.Blocks(theme=gr.themes.Glass())

class AppMain(CommunicationTranslator):

    def __init__(self):
        super().__init__()

    def main(self):
        self.build_translator_ui()
        self.build_scratchpad_ui()
        self.app = gr.TabbedInterface(
            [self.blocks_translator],
            ["Seamless Communication"],
            title="PavAI Translator",
            analytics_enabled=False
        )

    def launch(self):
        self.main()
        self.app.queue().launch()


if __name__ == "__main__":
    app = AppMain()
    app.launch()
