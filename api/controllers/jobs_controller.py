import pickle
import os
from flask import Blueprint, request, jsonify, make_response
from flask_restx import Api, Resource, fields, Namespace

# Create a Blueprint for the jobs controller
jobs_bp = Blueprint('jobs', __name__)
jobs_api = Api(jobs_bp, version='1.0', title='Jobs API',
               description='API for managing jobs')

# Define a namespace for jobs
jobs_ns = Namespace('jobs', description='Job operations')

# Define the model for a job
job_model = jobs_ns.model('Job', {
    'id': fields.Integer(required=True, description='The unique identifier of the job'),
    'description': fields.String(required=True, description='Description of the job'),
    'status': fields.String(description='Status of the job')
})


# File to persist data
DATA_FILE = 'jobs.pkl'


def load_jobs():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as file:
            return pickle.load(file)
    else:
        # Create an empty file if it doesn't exist
        with open(DATA_FILE, 'wb') as file:
            pickle.dump({}, file)
        return {}


def save_jobs(jobs):
    with open(DATA_FILE, 'wb') as file:
        pickle.dump(jobs, file)


# Load initial data
jobs_store = load_jobs()


@jobs_ns.route('/')
class JobList(Resource):
    def get(self):
        """List all jobs"""
        jobs_store = load_jobs()
        return _corsify_actual_response(jsonify(list(jobs_store.values())))

    @jobs_ns.expect(job_model)
    def post(self):
        """Create a new job"""
        if request.method == "OPTIONS":  # CORS preflight
            return _build_cors_preflight_response()

        job = request.json
        job_id = job['id']
        if job_id in jobs_store:
            return {'message': 'Job already exists'}, 400
        jobs_store[job_id] = job
        save_jobs(jobs_store)  # Save to file
        return _corsify_actual_response(jsonify(job))


@jobs_ns.route('/<int:job_id>')
class JobItem(Resource):
    def get(self, job_id):
        """Get a job by ID"""
        job = jobs_store.get(job_id)
        if job is None:
            return {'message': 'Job not found'}, 404
        return _corsify_actual_response(jsonify(job))

    @jobs_ns.expect(job_model)
    def put(self, job_id):
        """Update a job by ID"""
        if request.method == "OPTIONS":  # CORS preflight
            return _build_cors_preflight_response()
        job = request.json
        if job_id not in jobs_store:
            return {'message': 'Job not found'}, 404
        jobs_store[job_id] = job
        save_jobs(jobs_store)  # Save to file
        return _corsify_actual_response(jsonify(job))

    def delete(self, job_id):
        """Delete a job by ID"""
        if request.method == "OPTIONS":  # CORS preflight
            return _build_cors_preflight_response()
        if job_id not in jobs_store:
            return {'message': 'Job not found'}, 404
        del jobs_store[job_id]
        save_jobs(jobs_store)  # Save to file
        return _corsify_actual_response(jsonify({'message': 'Job deleted'}))


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
