import gradio as gr
import pavai.translator.filedata as filedata

class ScratchPad:
    def build_scratchpad_ui(self):
        self.blocks_scratchpad = gr.Blocks(analytics_enabled=False)
        with self.blocks_scratchpad:
            with gr.Accordion("Important tasks to complete", open=True):
                box_notepad = gr.TextArea(
                    lines=3,
                    info="write single page tasks.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to tasks.txt!")
                    return filedata.save_text_file("tasks.txt", text_notes)

                def load_notespad(filename: str = "tasks.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
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

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to notes.txt!")
                    return filedata.save_text_file("notes.txt", text_notes)

                def load_notespad(filename: str = "notes.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
            with gr.Accordion("Todos", open=False):
                box_notepad = gr.TextArea(
                    lines=3,
                    info="write single page of Todos.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to todos.txt!")
                    return filedata.save_text_file("todos.txt", text_notes)

                def load_notespad(filename: str = "todos.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
            with gr.Accordion("Reminders", open=False):
                box_notepad = gr.TextArea(
                    lines=3,
                    info="write single page of Todos.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to reminder.txt!")
                    return filedata.save_text_file("reminder.txt", text_notes)

                def load_notespad(filename: str = "reminder.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
        return self.blocks_scratchpad

class AppMain(ScratchPad):

    def __init__(self):
        super().__init__()

    def main(self):
        self.build_translator_ui()
        self.build_scratchpad_ui()
        self.app = gr.TabbedInterface(
            [self.blocks_scratchpad],
            ["Scratch Pad"],
            title="PavAI Scratchpad",
            analytics_enabled=False
        )

    def launch(self):
        self.main()
        self.app.queue().launch()


if __name__ == "__main__":
    app = AppMain()
    app.launch()
