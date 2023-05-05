import os
import pickle
from typing import Type

import pandas as pd
import ipywidgets as w
from ipyevents import Event
from IPython.display import display

from anomaly_detector import *

passwords = [
    "hello_World.219",
    "mouse.Keyboard&75",
    "insectSOUTHERN",
    "heaven-strength59",
    "company+INCREASE+10",
    "four+LOVE+17",
    "fence:PLANT:16",
    "cloud|EUROPE|18",
    "broke.WALES.79",
    "there?SERVE?37",
    "chart&MINUTE&59",
    "delight+GALAXY+25",
    "notice%MAIN%26",
    "dance+INCHES+38",
    "distance&GROUP&53",
    "arizona=RUSSIA=93",
    "bottle@WORN@47",
    "speak:RIVER:91",
    "charge-FINISHED-99",
    "after_LADY_38",
]

anomaly_detector_classes: dict[str, Type[AbstractAnomalyDetector]] = {
    'euclidean': EuclideanAnomalyDetector,
    'nn_mahalanobis': NearestNeighborMahalanobisAnomalyDetector,
    'mahalanobis': MahalanobisAnomalyDetector,
    'manhattan': ManhattanAnomalyDetector,
    'manhattan_scaled': ManhattanScaledAnomalyDetector,
    'one_class_svm': OneClassSvmAnomalyDetector,
    'outlier_count': OutlierCountAnomalyDetector
}


class DetectorContainer:
    def __init__(self, detector_class: Type[AbstractAnomalyDetector]):
        self.detector = detector_class()
        self.training_data: pd.DataFrame = pd.DataFrame()
        self.scores: list[float] = []


class User:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

        self.detectors: dict[str, DetectorContainer] = {
            detector_name: DetectorContainer(detector_class)
            for detector_name, detector_class in anomaly_detector_classes.items()
        }
        self.retraining_enabled: list[bool] = []


users: dict[str, User] = {}

if os.path.isfile('app_state.pkl'):
    with open('app_state.pkl', 'rb') as f:
        users = pickle.load(f)
    saving_status_str = 'Loaded from "app_state.pkl".'
else:
    for i in range(len(passwords)):
        username = f"user #{i + 1}"
        password = passwords[i]
        users[username] = User(username, password)
    saving_status_str = "No 'app_state.pkl' found, using defaults."

current_user = users['user #1']

user_dropdown = w.Dropdown(
    options=users.keys(),
    description="User Profile:"
)

password_hint_label = w.Label(f"Password Hint: {current_user.password}")
password_entry_text_widget = w.Text(
    placeholder='Enter your password here'
)
password_entry_text_widget.layout.border = '1px solid white'
enable_retraining_checkbox = w.Checkbox(
    value=True, description='Re-train detectors after scoring', indent=False
)

WAITING_FOR_PASSWORD_STATUS = 'Press "Enter" to finish entering the password.'
SCORING_IN_PROGRESS_STATUS = 'Scoring your keystroke dynamics data...'
AUTH_PASSED_STATUS = 'You are authenticated. Re-training detector...'
RETRAINING_COMPLETED_STATUS = 'You are authenticated. Re-training completed.'
INCORRECT_PASSWORD_STATUS = "Password didn't match! Try again."
AUTH_FAILED_STATUS = 'Access denied! Try again.'

save_to_file_button = w.Button(
    description='Save everything to file'
)
saving_status_label = w.Label(saving_status_str)
status_label = w.Label(WAITING_FOR_PASSWORD_STATUS)
detector_scores_dataframe_output = w.Output(
    layout={'height': '415px', 'overflow': 'scroll', 'border': 'solid 1px black'}
)
entered_keys_dataframe_output = w.Output(
    layout={'height': '150px', 'overflow': 'scroll', 'border': 'solid 1px black'}
)

left_side_vbox = w.VBox([
    user_dropdown,
    password_hint_label,
    password_entry_text_widget,
    enable_retraining_checkbox,
    w.HBox([save_to_file_button, saving_status_label]),
    status_label
])

right_side_vbox = w.VBox([
    w.Label('Entered Keys:'),
    entered_keys_dataframe_output
])

curr_feat_vector = []
last_timestamp = 0
last_entered_key = ''


def create_scores_df():
    result_df = pd.DataFrame()
    for detector_name in anomaly_detector_classes.keys():
        curr_detector = current_user.detectors[detector_name]
        result_df[detector_name] = curr_detector.scores
    result_df['re-trained'] = current_user.retraining_enabled
    # To make the latest scores appear first.
    return result_df[::-1]


def create_entered_keys_df():
    entered_keys, durations, interval_types = [], [], []
    for feat_tuple in curr_feat_vector:
        entered_keys.append(feat_tuple[0])
        durations.append(feat_tuple[1])
        interval_types.append(feat_tuple[2])
    result_df = pd.DataFrame()
    result_df['Key'] = entered_keys
    result_df['Duration'] = durations
    result_df['Type'] = interval_types
    return result_df[::-1]


with detector_scores_dataframe_output:
    display(create_scores_df())
with entered_keys_dataframe_output:
    display(create_entered_keys_df())

box_widget = w.HBox([left_side_vbox, right_side_vbox])

event_widget = Event(source=box_widget, watched_events=['keydown', 'keyup'])


def on_save_to_file_button_clicked(_b):
    saving_status_label.value = "Saving..."
    try:
        with open('app_state.pkl', 'wb') as f:
            pickle.dump(users, f)
    except Exception as e:
        saving_status_label.value = str(e)
        raise e
    saving_status_label.value = "Saved successfully."


save_to_file_button.on_click(on_save_to_file_button_clicked)


def user_dropdown_change(change):
    global curr_feat_vector
    global last_timestamp
    global last_entered_key
    global current_user

    new_value = change.new
    current_user = users[new_value]
    password_hint_label.value = f'Password Hint: {current_user.password}'
    password_entry_text_widget.disabled = False
    password_entry_text_widget.value = ''
    curr_feat_vector = []
    last_timestamp = 0
    last_entered_key = ''

    detector_scores_dataframe_output.clear_output()
    with detector_scores_dataframe_output:
        display(create_scores_df())
    entered_keys_dataframe_output.clear_output()
    with entered_keys_dataframe_output:
        display(create_entered_keys_df())


user_dropdown.observe(user_dropdown_change, names='value')


def handle_event(event):
    global curr_feat_vector
    global last_timestamp
    global last_entered_key

    key_value = event['key']
    event_type = event['event']
    keydown_event = 'keydown'
    curr_timestamp = event['timeStamp']

    if event_type == keydown_event:
        if len(key_value) == 1:
            password_entry_text_widget.value += key_value
        elif key_value == 'Backspace':
            password_entry_text_widget.value = password_entry_text_widget.value[:-1]
        elif key_value == 'Enter':
            password_entry_text_widget.disabled = True

            if password_entry_text_widget.value == current_user.password:
                status_label.value = SCORING_IN_PROGRESS_STATUS
                new_training_vector = pd.DataFrame(item[1] for item in curr_feat_vector)
                for curr_anomaly_detector in current_user.detectors.values():
                    score = 0
                    if curr_anomaly_detector.scores:
                        score = curr_anomaly_detector.detector.score(new_training_vector)['score'].iloc[0]

                    if enable_retraining_checkbox.value:
                        if not curr_anomaly_detector.training_data.size:
                            curr_anomaly_detector.training_data = new_training_vector.copy()
                        else:
                            curr_anomaly_detector.training_data = pd.concat([
                                curr_anomaly_detector.training_data,
                                new_training_vector
                            ])
                        curr_anomaly_detector.detector.train(
                            curr_anomaly_detector.training_data
                        )
                    curr_anomaly_detector.scores.append(round(score, 2))
                current_user.retraining_enabled.append(enable_retraining_checkbox.value)
                if enable_retraining_checkbox.value:
                    status_label.value = RETRAINING_COMPLETED_STATUS
                else:
                    status_label.value = 'You are authenticated'
            else:
                status_label.value = INCORRECT_PASSWORD_STATUS

            password_entry_text_widget.disabled = False
            password_entry_text_widget.value = ''
            curr_feat_vector = []
            last_timestamp = 0
            last_entered_key = ''
            detector_scores_dataframe_output.clear_output()
            with detector_scores_dataframe_output:
                display(create_scores_df())

    if not current_user.password.startswith(password_entry_text_widget.value):
        password_entry_text_widget.layout.border = '1px solid red'
    else:
        password_entry_text_widget.layout.border = '1px solid white'
        if password_entry_text_widget.value and len(key_value) == 1:
            if last_timestamp != 0:
                curr_feat_vector.append((
                    last_entered_key + key_value if event_type == keydown_event else key_value,
                    curr_timestamp - last_timestamp,
                    'UD' if event_type == keydown_event else 'H'
                ))
            last_timestamp = curr_timestamp
            last_entered_key = key_value

    if password_entry_text_widget.value:
        entered_keys_dataframe_output.clear_output()
        with entered_keys_dataframe_output:
            display(create_entered_keys_df())


event_widget.on_dom_event(handle_event)

scores_box = w.VBox(
    [
        w.Label('Scores:'),
        detector_scores_dataframe_output
    ]
)

display(
    box_widget,
    scores_box,
)
