# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from urllib.parse import urlparse, parse_qs

from src.dataset import Dataset
from src.trainer import Trainer
from src.predictor import Predictor

hostName = "localhost"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.get_params(self.path)
        result = self.get_data(self.par_text)
        self.wfile.write(bytes("<html><head><title>Translator Api</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.par_action, "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.par_text, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes(f"<p>Translation is: {result}</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def get_params(self, path) :
        url = urlparse(path)
        params = parse_qs(url.query)
        self.par_action = params['action'][0]
        self.par_text = params['text'][0]

    def get_data(self, text) : 
            match self.par_action:
                case "predict":
                    pr = Predictor()
                    while(1):
                        if text == 'quit':
                            break
                        prediction = pr.predict(text)
                        return prediction
                case "train":
                    tr = Trainer()
                    tr.train()
                case "create_dataset":
                    ds = Dataset()
                    ds.build()
                case _:
                    print("There is no such action")


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")