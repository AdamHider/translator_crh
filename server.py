import sys
sys.path.insert(0, './src/')

from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort

from flask_cors import CORS, cross_origin

#from src.trainer import Trainer
from src.predictor import Predictor

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py
cors = CORS(app)
# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['CORS_HEADERS'] = 'Content-Type'
predictor = Predictor()

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')


@app.route('/about')
@cross_origin()
def about():
    return 'About Us'


@app.route('/predictor/get', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def translate_eng():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_preflight_response()
        
    if request.method == 'POST':
        if not request.json or 'text' not in request.json or 'level' not in request.json or 'target_lang' not in request.json:
            abort(400)
        text = request.json['text']
        level = request.json['level']
        target_lang = request.json['target_lang']
    else:
        text = request.args.get('text')
        level = request.args.get('level')
        target_lang = request.args.get('target_lang')

    target_text = text
    if level == 'char' and target_lang == 'french':
        target_text = predictor.predict(text)
        
    return jsonify({
        'sentence': text,
        'translated': target_text,
        'target_lang': target_lang,
        'level': level
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    app.run(debug=True)


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    response.headers.add('Access-Control-Allow-Credentials', '*')
    return response

if __name__ == '__main__':
    main()
