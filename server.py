from helper import url_to_image
from inference import SegApp
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource

seg_app = SegApp()
seg_app.spin()
app = Flask(__name__)
api = Api(app)
img_url = ""


class WebServer(Resource):
    def get(self):
        return {'status': True}

    def post(self):
        json_data = request.get_json(force=True)
        img_url = json_data['img_url']
        img = url_to_image(img_url)
        _ = seg_app.process(img)
        prediction = seg_app.get_result()
        # You may wanna play some magic here.
        return jsonify(str(prediction))


api.add_resource(WebServer, '/segmentation')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=80)
