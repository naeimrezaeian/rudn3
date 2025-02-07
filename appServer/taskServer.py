import time
import database
import rec_analyzer
import os
import ffmpeg
import assistant
import warnings
warnings.filterwarnings("ignore")
db=database.Database("assistantdb")
PYANNOTE_AUTH_TOKEN="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"
GIGACHAT_API_KEY="YTQwODIyNjAtMDQ0YS00MGI5LTg0ZmQtMjRlMmMzNTA5NzBhOjgxYmJmNTdhLTIxYTYtNGNlMi04YWFkLTBkMGIzMzM4ODA4MA=="

def convert_mp4_to_mp3(input_file, output_file):
    try:
        ffmpeg.input(input_file).output(output_file, acodec='libmp3lame', audio_bitrate='192k').run()
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr.decode()}")
def process_audio(filename):
     folder="uploads"
     full_path = os.path.join(folder, filename)
     file_extension=filename.split(".")[1]
     file_name=filename.split(".")[0]
     if file_extension.lower()=="mp4":
         mp3_filename=f"{file_name}.mp3"         
         convert_mp4_to_mp3(full_path,os.path.join(folder, mp3_filename))
         full_path=os.path.join(folder, mp3_filename)
         

     analyzer = rec_analyzer.LectureHelper(full_path, GIGACHAT_API_KEY, PYANNOTE_AUTH_TOKEN,"123")
     
     print(f"Processing file: {filename}")  
     return analyzer.get_results()


def main():
    while True:
        record=db.get_tasks()
        if record:
            id=record["_id"]
            file_path = record["filename"]
            try:
                 results=process_audio(file_path)
                 db.update_task(record["_id"],results,1)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                db.update_task(record["_id"],2)
        
if __name__ == "__main__":
    main()









