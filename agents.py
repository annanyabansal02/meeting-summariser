from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Initialize Mistral via Ollama
llm = OllamaLLM(model="mistral")

# ─── AGENT 1: SUMMARISER ───
summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an expert meeting summariser.
Read the following meeting transcript and provide a clear, 
concise summary in 4-5 sentences covering the main topics discussed.

Transcript:
{transcript}

Summary:
"""
)

summary_chain = summary_prompt | llm

# ─── AGENT 2: ACTION ITEM EXTRACTOR ───
action_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an expert at extracting action items from meeting transcripts.
Read the following transcript and extract ALL action items.

For each action item provide:
- Task: what needs to be done
- Owner: who is responsible
- Deadline: when it needs to be done (if mentioned)

Format your response as a numbered list like this:
1. Task: [task description] | Owner: [name] | Deadline: [deadline]

Transcript:
{transcript}

Action Items:
"""
)

action_chain = action_prompt | llm

# ─── AGENT 3: DECISION EXTRACTOR ───
decision_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an expert at identifying key decisions from meeting transcripts.
Read the following transcript and extract ALL decisions that were made.

Format your response as a numbered list like this:
1. [Decision made]

Transcript:
{transcript}

Decisions Made:
"""
)

decision_chain = decision_prompt | llm

# ─── AGENT 4: KEY POINTS EXTRACTOR ───
keypoints_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an expert at identifying key discussion points from meetings.
Read the following transcript and extract the top 5 key points discussed.

Format as a numbered list:
1. [Key point]

Transcript:
{transcript}

Key Points:
"""
)

keypoints_chain = keypoints_prompt | llm

def analyse_meeting(transcript):
    """Run all 4 agents on the transcript"""
    print("Agent 1: Generating summary...")
    summary = summary_chain.invoke({"transcript": transcript})

    print("Agent 2: Extracting action items...")
    action_items = action_chain.invoke({"transcript": transcript})

    print("Agent 3: Extracting decisions...")
    decisions = decision_chain.invoke({"transcript": transcript})

    print("Agent 4: Extracting key points...")
    key_points = keypoints_chain.invoke({"transcript": transcript})

    return {
        "summary": summary.strip(),
        "action_items": action_items.strip(),
        "decisions": decisions.strip(),
        "key_points": key_points.strip()
    }


# Test it
if __name__ == "__main__":
    with open("sample_meeting.txt", "r") as f:
        transcript = f.read()

    results = analyse_meeting(transcript)

    print("\\n" + "="*50)
    print("SUMMARY:")
    print(results["summary"])
    print("\\n" + "="*50)
    print("ACTION ITEMS:")
    print(results["action_items"])
    print("\\n" + "="*50)
    print("DECISIONS:")
    print(results["decisions"])
    print("\\n" + "="*50)
    print("KEY POINTS:")
    print(results["key_points"])