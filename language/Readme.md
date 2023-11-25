# JSON Language Translation

This script is for translating JSON files used in this proejct. It utilizes the Google Cloud Translation API to automatically translate text within JSON files from one language to another.

## Getting Started

### Prerequisites
- Google Cloud account and the Google Cloud CLI
- Enabled Google Cloud Translation API
- Google Cloud service account key

### Initial Setup
1. **Re-install Required Python Packages**:
   - Install the necessary package using the `requirements_versions.txt` file.
     ```
     pip install -r requirements_versions.txt
     ```

2. **Google Cloud Translation API Setup**:
   - Ensure you have a Google Cloud account and have activated the Google Cloud Translation API in your Google Cloud project.
   - Install and initialize the Google Cloud CLI following the instructions [here](https://cloud.google.com/sdk/docs/install).
   - For more detailed instructions on providing credentials for local development, refer to the [official documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev).

### Usage in Web Projects
- Place the JSON file that requires translation in your web project's designated directory.
- Run the script specifying the input file, output file, and the target language code.
