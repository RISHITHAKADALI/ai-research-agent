import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from duckduckgo_search import DDGS
from groq import Groq

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def search_web(query):
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=5))
    context = ""
    for r in results:
        context += f"Title: {r.get('title','')}\nContent: {r.get('body','')}\n\n"
    return context

def run_agent(query):
    context = search_web(query)
    prompt = f"""You are an expert research assistant.
Question: {query}
Research Context: {context}
Provide:
1. Summary
2. Key findings
3. Important insights
4. Final conclusion"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

@app.route('/')
def home():
    return "AI Research Agent is running! ✅"

@app.route('/run_agent', methods=['POST'])
def agent_endpoint():
    try:
        data = request.get_json()
        query = data.get("query", "")
        answer = run_agent(query)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
