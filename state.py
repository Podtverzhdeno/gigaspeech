from langgraph.graph import MessagesState

class SpeechWriterState(MessagesState):
    speech_topic: str
    time_to_speak: str
    speaker_bio: str
    result_speech: str
    critique: list[str]
    retriever_docs: str
    speech_structure: str
    speech_tech_spec: str
    relevant_quotes_309: list[str]
