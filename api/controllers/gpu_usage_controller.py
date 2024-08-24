from flask_restx import Api, Resource, fields, Namespace
from flask import Flask, jsonify, render_template, send_from_directory, Blueprint, request, jsonify, make_response
import psutil
import GPUtil
import time
# Create a Blueprint for the gpu_usage controller
gpu_usage_bp = Blueprint('gpu_usage', __name__)
gpu_usage_api = Api(gpu_usage_bp, version='1.0', title='gpu_usage API',
               description='API for managing gpu_usage')

# Define a namespace for gpu_usage
gpu_usage_ns = Namespace('gpu_usage', description='gpu usage operations')

# Define the model for a gpu
gpu_model = gpu_usage_ns.model('gpu_usage', {
    'id': fields.Integer(required=True, description='The unique identifier of the gpu'),
    'description': fields.String(required=True, description='Description of the gpu'),
    'status': fields.String(description='Status of the gpu')
})


# Cache for system usage data
cache = {
    'timestamp': 0,
    'data': {
        'cpu': 0,
        'memory': 0,
        'gpu': 0,
        'vram': 0,
        'hdd': 0
    }
}
CACHE_DURATION = 1  # Cache duration in seconds

@gpu_usage_ns.route('/')
class GPUInfo(Resource):
    def get(self):
        if request.method == "OPTIONS":  # CORS preflight
                return _build_cors_preflight_response()

        current_time = time.time()

        # Check if the cache is still valid
        if current_time - cache['timestamp'] < CACHE_DURATION:
            return _corsify_actual_response(jsonify(cache['data']))

        try:
            # Get CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0)

            # Get Memory utilization
            mem = psutil.virtual_memory()
            mem_percent = mem.percent

            # Get GPU utilization (considering only the first GPU)
            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].load * 100 if gpus else 0

            # Get VRAM usage (considering only the first GPU)
            vram_usage = 0
            if gpus:
                used = gpus[0].memoryUsed
                total = gpus[0].memoryTotal
                vram_usage = (used / total) * 100

            # Get HDD usage (assuming usage of the primary disk)
            hdd = psutil.disk_usage('/')
            hdd_percent = hdd.percent

            # Update the cache
            cache['data'] = {
                'cpu': cpu_percent,
                'memory': mem_percent,
                'gpu': gpu_percent,
                'vram': vram_usage,  # Convert bytes to MB
                'hdd': hdd_percent
            }
            cache['timestamp'] = current_time

            return _corsify_actual_response(jsonify(cache['data']))
        except Exception as e:
            return _corsify_actual_response(jsonify({'error': str(e)}), 500)


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response