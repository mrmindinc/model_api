from flask import request
from flask_api import FlaskAPI, status
import json
from text_check import ExtractIntent

app = FlaskAPI(__name__)

@app.route("/api/response", methods=['GET', 'POST'])
def api():
    # chat message
    if request.method == 'POST':
        generat_intent = ExtractIntent()
        
        _data = request.data
        if isinstance(_data, str):
            _data = json.loads(_data)

        response = generat_intent.nlp_process(_data)

        return json.dumps(response, ensure_ascii=False), status.HTTP_200_OK

@app.route("/api/response/sculpture", methods=['GET', 'POST'])
def api2():
    # chat message
    if request.method == 'POST':
        generat_intent = ExtractIntent()
        
        _data = request.data
        if isinstance(_data, str):
            _data = json.loads(_data)

        response = generat_intent.nlp_process_sculpture(_data)

        return json.dumps(response, ensure_ascii=False), status.HTTP_200_OK

if __name__ == '__main__':

    default_host = "0.0.0.0"
    default_port = "10733"

    import os
    os.environ['FLASK_ENV'] = "development"

    import optparse
    parser = optparse.OptionParser()

    parser.add_option("-H", "--host",
                      help="Hostname of the Flask app " + \
                           "[default %s]" % default_host,
                      default=default_host)

    parser.add_option("-P", "--port",
                      help="Port for the Flask app " + \
                           "[default %s]" % default_port,
                      default=default_port)

    options, _ = parser.parse_args()

    app.run(
        debug=True,
        host=options.host,
        port=int(options.port)
    )