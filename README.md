## Installation Instructions

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\\Scripts\\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory and add the following(example):
      ```env
      DATABASE_URL=<postgresql://postgres:password@localhost:5432/rasa_prod>
      OPENAI_API_KEY=<YOUR KEY>
      MODEL_TYPE="ollama" or "openai"
      ```

5. **Run the Rasa actions server**:
    ```bash
    python -m rasa run actions
    ```

6. **Run the Rasa shell**:
    ```bash
    python -m rasa shell
    ```