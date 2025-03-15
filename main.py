import json
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage


app = Flask(__name__)


with open("experts.json", "r") as f:
    experts_data = json.load(f)

with open("services.json", "r") as f:
    services_data = json.load(f)


chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def extract_keywords(query):
    """Extracts keywords and generates a concise explanation (100 words) based on the user query."""
    if not query:
        return [], ""

    prompt = [
        SystemMessage(content="You are an AI that provides concise explanations and suggestions based on user queries. Limit your response to 100 words."),
        HumanMessage(content=f"Provide a concise explanation and suggestions for this query: '{query}'. Limit your response to 100 words.")
    ]
    response = chat_model.invoke(prompt)
    
    if response.content:
       
        keywords = query.lower().split()
       
        explanation = " ".join(response.content.split()[:100])
        return keywords, explanation
    else:
        return [], "Unable to generate a response. Please try again."

def search_experts_and_services(keywords):
    """Find matching experts and services based on extracted keywords."""
    matching_experts = []
    matching_services = []
    
    for expert in experts_data:
        
        expertise = expert.get("expertise", "").lower()
        if any(keyword.lower() in expertise for keyword in keywords):
            matching_experts.append(expert)
    
    for service in services_data:
       
        name = service.get("name", "").lower()
        if any(keyword.lower() in name for keyword in keywords):
            matching_services.append(service)
    
    return matching_experts, matching_services


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    
    query = request.form.get("query", "")
    print(f"Received query: {query}")  

   
    keywords, explanation = extract_keywords(query)
    print(f"Extracted keywords: {keywords}") 
    print(f"Generated explanation: {explanation}")  

   
    experts, services = search_experts_and_services(keywords)
    
   
    if not experts and not services:
        explanation = (
            "No specific experts or services found for your query. "
            "Please explain more about the specific domain or topic you are looking for. "
            "For example, are you interested in residential real estate, commercial real estate, or property management?"
        )
    
    
    if request.headers.get("Content-Type") == "application/json" or request.args.get("format") == "json":
        
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


if __name__ == "__main__":
    app.run(debug=True, port=5050)