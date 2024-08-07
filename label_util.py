from ipywidgets import Output, Button, Layout, HBox, VBox
from IPython.display import Image, display, clear_output, Markdown, HTML
from itertools import zip_longest

# input dataset with nodes
# iterate over nodes
# sample random model
# sample softmaxed model second

# sample randomly


class LabelingTool:
    def __init__(
        self, dataset, assessor_id, image_path,
        save2path=None,
        width=300, height=300, ignore_all_labeled=False, 
        img_type='.jpg'
    ):
        self.dataset = dataset
        self.indices = self.dataset.index
        self.use_images = True
        self.ignore_all_labeled = ignore_all_labeled
        self.len = len(self.dataset)
        self.assessor_id = assessor_id

        self.width = width
        self.height = height
        self.save2path = save2path

        self.label_column = f'assessor_{assessor_id}'

        if self.label_column not in self.dataset:
            self.dataset[self.label_column] = None

        assert 'model_1' in self.dataset.columns and 'model_2' in self.dataset.columns, "No models defined for each pair"

        try:
            self.position = self.dataset.index[self.dataset[self.label_column].notna()][-1]
        except IndexError:
            self.position = 0

        self.image_path = image_path
        self.img_type = img_type

        forward_button = Button(description="Next ➡️")
        forward_button.on_click(self._next_image)

        backward_button = Button(description="⬅️ Prev")
        backward_button.on_click(self._prev_image)

        labels = ['Model A is Better', 'Model B is Better', 'Tie', 'Both are bad']
        labels_buttons = [Button(description=label) for label in labels]
        for button in labels_buttons:
            button.on_click(self._save_label)

        self.image_1_frame = Output()
        self.image_2_frame = Output()
        self.utter_frame = Output(layout=Layout(max_width="400px"))
        self.label_frame = Output()
        self.position_frame = Output()

        stop_button = Button(description='Stop ⛔')
        stop_button.on_click(self._stop)
        remove_button = Button(description='Remove label 🗑️')
        remove_button.on_click(self._remove_label)

        self.navigation_box = HBox([backward_button, forward_button])
        self.labels_box = HBox(labels_buttons)
        self.action_box = HBox([stop_button, remove_button])

        self.control_box = VBox([
            self.utter_frame, self.position_frame, self.label_frame,
            self.navigation_box, self.labels_box,
            self.action_box
        ], layout=Layout(align_items='center'))

        self.image_box = HBox([self.image_1_frame, self.image_2_frame], layout=Layout(align_items='center'))

        self.widgets = VBox([
            self.image_box,
            self.control_box
        ], layout=Layout(align_items='center', border='100px'))


        self.label2value = {'Model A is Better': 1, 'Model B is Better': 2, "Tie": 3, "Both are bad": 4}

    def _remove_label(self, button):
        idx = self.indices[self.position]
        self.dataset[self.label_column][idx] = None
        with self.label_frame:
            clear_output(wait=True)
            display(Markdown(str(None)))

    def close_and_save(self, msg):
        self.widgets.close()
        clear_output(wait=True)
        print(msg)
        if self.save2path is not None:
            self.dataset.to_csv(self.save2path)



    def _display_sample(self):
        if (not self.ignore_all_labeled and
                all(self.dataset[self.label_column][idx] is not None
                    for idx in self.indices)):
            self.close_and_save("All samples are labeled!")
            return

        if self.position < 0:
            self.position += 1
        if self.position >= len(self.indices):
            self.position -= 1
        idx = self.indices[self.position]
        item = self.dataset.loc[idx]

        path2image1 = f'{self.image_path}/{item["model_1"]}/{idx}{self.img_type}'
        with self.image_1_frame:
            clear_output(wait=True)
            display(Image(path2image1, height=300))

        path2image2 = f'{self.image_path}/{item["model_2"]}/{idx}{self.img_type}'
        with self.image_2_frame:
            clear_output(wait=True)
            display(Image(path2image2, height=300))

        with self.utter_frame:
            clear_output(wait=True)
            display(HTML(item['core_lemma']))

        with self.position_frame:
            clear_output(wait=True)
            display(Markdown('current position: ' + str(self.position) + '/' + str(self.len)))


        with self.label_frame:
            clear_output(wait=True)
            display(Markdown(str(item[self.label_column])))



    def _next_image(self, button=None):
        self.position += 1
        self.dataset.to_csv(self.save2path)
        self._display_sample()

    def _prev_image(self, button=None):
        self.position -= 1
        self._display_sample()

    def _save_label(self, button):
        idx = self.indices[self.position]
        self.dataset[self.label_column][idx] = self.label2value.get(button.description)
        self._next_image()

    def _stop(self, button):
        if self.dataset[self.label_column].isna().sum() == 0:
            msg = "All samples are labeled!"
        else:
            msg = "Labeling is stopped!"
        self.close_and_save(msg)

    def label_samples(self):
        display(self.widgets)
        self._display_sample()