# SHL Conversational Assessment Recommender

## About the Project
This project is a conversational assistant that helps hiring managers and recruiters select the right assessments from the SHL product catalog. When a user describes a job role or the skills they want to test, the assistant searches the SHL catalog and recommends a list of relevant assessments.

## Features
- Ask clarifying questions if the user request is too vague.
- Recommend a list of 1 to 10 matching assessments based on the job requirements.
- Refine the recommendation list when the user asks to add or remove specific tests.
- Refuse out-of-scope requests, such as legal or HR strategy questions.
- Handle a conversation limit of 8 turns.

## Technologies Used
- FastAPI: Used to build the web API endpoints because it is fast and automatically handles request validation.
- Python: The primary programming language used for the backend logic.
- Gemini: Used as the language model to generate natural responses and recommendations.
- Scikit-Learn: Used to build a TF-IDF text search index because it runs locally with very low memory, preventing out-of-memory issues on free hosting.
- FAISS: Used to structure the search index files.

## Project Structure
- llm: Contains the code for constructing prompts and communicating with the Gemini API.
- retrieval: Contains the search index logic that finds relevant assessments from the catalog.
- scraper: Contains the catalog database JSON file and the script that formats the catalog data.

## How It Works
1. The user sends a chat message to the API.
2. The API processes the conversation history and constructs a search query.
3. The retrieval module searches the catalog database using TF-IDF similarity to find the most relevant assessments.
4. The conversation history and the retrieved assessments are sent to the Gemini LLM.
5. The LLM generates a JSON response containing a text reply and a list of recommended assessments.

## API Endpoints
- GET /health: Returns a simple status indicating if the server is running.
- POST /chat: Receives the conversation history and returns the assistant reply with any recommendations.

## Notes
- The catalog database is stored locally in catalog.json. The search index is built from this file during deployment.
- The conversation history is ingested fully on every request because the API is completely stateless.
- Recommendation URLs are verified against the catalog to prevent the language model from generating broken links.
