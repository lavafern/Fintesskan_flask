from flask import Flask,render_template,Response,request
from bicep_curl_function import generate_frames
import os
from werkzeug.utils import secure_filename

app=Flask(__name__)


#create folder path of the uploaded video
base_directory = os.path.dirname(os.path.abspath(__file__))
upload_folder=os.path.join(base_directory, 'static', 'upload')

           

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video',methods=['GET','POST'])
def video():
    if request.method=='POST':
        video_file=request.files['video']
        filename = secure_filename(video_file.filename)
        vid_path=os.path.join(upload_folder, filename)
        
        video_file.save(vid_path)
        
        
    return Response(generate_frames(vid_path),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False)





