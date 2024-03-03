import gradio as gr
import pavai.translator.filedata as filedata

class CustomVoiceEditor:
    def build_audio_ui(self):
        self.blocks_voice_editor = gr.Blocks(analytics_enabled=False)
        with self.blocks_voice_editor:
            with gr.Accordion("Record Your Custom Voice", open=True):
                custom_audio_input = gr.Audio(scale=2, sources=["microphone", "upload"], 
                                                type="filepath", label="press [record] to start",
                                                show_download_button=True, 
                                                visible=True,
                                                format="wav", max_length=30,
                                                waveform_options={"waveform_progress_color": "green", "waveform_progress_color": "green"})

                with gr.Row():
                    with gr.Column(scale=3):
                        text_custom_audio_name = gr.Textbox(label="short name for the audio")
                    with gr.Column(scale=3):
                        btn_save_custom_audio = gr.Button(size="sm", value="Save")
                        btn_load_custom_audio = gr.Button(size="sm", value="Load")

                ## function: audio
                def save_audio(audiofile, audioname):
                    gr.Info("saved audio to resources/models/styletts2/reference_audio")
                    return filedata.save_text_file("audioname", audioname)

                def load_audio(audiofile, audioname):
                    gr.Info("load audio to resources/models/styletts2/reference_audio")
                    #return filedata.save_text_file("audioname", audioname)

                btn_save_custom_audio.click(fn=save_audio, inputs=[custom_audio_input,text_custom_audio_name])
                btn_load_custom_audio.click(fn=load_audio, outputs=[custom_audio_input])
                ## update expert mode and speech style
            with gr.Accordion("Notes", open=False):
                box_notepad = gr.TextArea(
                    lines=5,
                    info="write single page quick notes.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")


        return self.blocks_voice_editor

class AppMain(CustomVoiceEditor):

    def __init__(self):
        super().__init__()

    def main(self):
        self.build_audio_ui()
        self.app = gr.TabbedInterface(
            [self.blocks_voice_editor],
            ["Audio Editor"],
            title="Pavai Audio Editor",
            analytics_enabled=False
        )

    def launch(self):
        self.main()
        self.app.queue().launch()


if __name__ == "__main__":
    app = AppMain()
    app.launch()
