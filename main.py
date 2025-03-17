import json
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Initialize Flask app
app = Flask(__name__)

# Load experts and services data
with open("experts.json", "r") as f:
    experts_data = json.load(f)

with open("services.json", "r") as f:
    services_data = json.load(f)

# Initialize Gemini model
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def extract_keywords(query):
    """Extracts keywords and generates a concise explanation (100 words) based on the user query."""
    if not query:
        return [], ""

    prompt = [
        SystemMessage(content="You are an AI that extracts key topics and related terms from user queries."),
        HumanMessage(content=f"Extract the main keywords and related topics from this query: '{query}'.")
    ]
    response = chat_model.invoke(prompt)

    if response.content:
        keywords = [word.strip().lower() for word in response.content.split(",")]
        explanation = " ".join(response.content.split()[:100])
        return keywords, explanation
    else:
        return [], "Unable to generate a response. Please try again."

def search_experts_and_services(keywords):
    """Find matching experts and services based on extracted keywords."""
    matching_experts = []
    matching_services = []

    general_terms = ["coding", "development", "technology", "software"]
    
    for expert in experts_data:
        expertise = expert.get("expertise", "").lower()
        if any(keyword in expertise for keyword in keywords):
            matching_experts.append(expert)

    for service in services_data:
        name = service.get("name", "").lower()
        description = service.get("description", "").lower()

        # Match exact or related keywords
        if any(keyword in name or keyword in description for keyword in keywords):
            matching_services.append(service)

        # If the user searches for "coding", return all related services
        elif any(keyword in general_terms for keyword in keywords):
            matching_services.append(service)

    return matching_experts, matching_services

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Search endpoint
@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    print(f"Received query: {query}")

    keywords, explanation = extract_keywords(query)
    print(f"Extracted keywords: {keywords}")
    print(f"Generated explanation: {explanation}")

    experts, services = search_experts_and_services(keywords)

    if not experts and not services:
        explanation = (
            "No specific experts or services found for your query. "
            "Please clarify your request. Are you looking for software development, AI, cybersecurity, or another service?"
        )

    if request.headers.get("Accept") == "application/json" or request.args.get("format") == "json":
        return jsonify({
            "query": query,
            "keywords": keywords,
            "explanation": explanation,
            "experts": experts,
            "services": services
        })
    else:
        return render_template(
            "results.html",
            query=query,
            keywords=keywords,
            explanation=explanation,
            experts=experts,
            services=services
        )

# Run the Flask app on port 5050
if __name__ == "__main__":
    app.run(debug=True, port=5050)
