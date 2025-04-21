import os
from tasks.detection import DetectAndAnnotate
from tasks.tracking import TrackAndAnnotate

system_directory = os.getcwd()
input_file_path = f"{system_directory}/RBK_TDT17/4_annotate_1min_bodo_start"
output_file_path = f"{system_directory}/tracked_output"


def run():
    process = TrackAndAnnotate(input_file_path=input_file_path, output_file_path=output_file_path)
    process.track_and_anotate()




run()
