
import os
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
import random

# Mock-Funktion zur Simulation einer Wetter-API
def get_weather(city: str):
    """Simuliert das Abrufen von Wetterdaten für eine bestimmte Stadt."""
    if "berlin" in city.lower():
        return f"Die Temperatur in Berlin ist {random.randint(10, 25)} Grad Celsius."
    elif "paris" in city.lower():
        return "In Paris scheint die Sonne."
    else:
        return f"Wetter für {city} konnte nicht abgerufen werden."

# 1. State Management: Definition des AgentState
class AgentState(TypedDict):
    """
    Repräsentiert den Zustand unseres Agenten.

    Attributes:
        messages: Der Verlauf der Nachrichten in der Konversation.
        validation_passes: Ein Zähler für erfolgreiche Validierungen.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    validation_passes: int

# 2. Agent-Knoten
def agent_node(state: AgentState):
    """
    Der zentrale Agenten-Knoten, der die Anfrage verarbeitet.
    Hier würde in der Praxis ein LLM aufgerufen.
    """
    print("---AGENTEN-KNOTEN---")
    last_message = state['messages'][-1]
    
    # Extrahiert die Stadt aus der Nachricht (vereinfachte Logik für das MVP)
    city = "unbekannt"
    if "KORREKTUR" in last_message.content:
        # Extrahiert die Stadt aus der Korrektur-Aufforderung
        parts = last_message.content.split("'")
        if len(parts) > 1:
            city = parts[1]
    else:
        # Extrahiert die Stadt aus der initialen Anfrage
        parts = last_message.content.split("für")
        if len(parts) > 1:
            city = parts[-1].strip().replace("?","")

    print(f"Stadt extrahiert: {city}")
    response_text = get_weather(city)
    
    new_message = HumanMessage(content=response_text, name="Agent")
    return {"messages": [new_message]}

# 3. Validator-Knoten
def validator_node(state: AgentState):
    """
    Validiert die Antwort des Agenten.
    Prüft, ob eine Temperaturangabe enthalten ist.
    """
    print("---VALIDATOR-KNOTEN---")
    last_message = state['messages'][-1]
    
    print(f"Validiere Antwort: '{last_message.content}'")
    
    if "grad" in last_message.content.lower():
        print("Validierung erfolgreich: Temperatur gefunden.")
        # Erhöhe den Zähler, um Erfolg zu signalisieren
        passes = state.get('validation_passes', 0) + 1
        return {"validation_passes": passes}
    else:
        print("Validierung fehlgeschlagen: Keine Temperaturangabe.")
        # Der Zähler wird nicht erhöht
        return {}

# NEU: Correction-Knoten
def correction_node(state: AgentState):
    """
    Erstellt eine Korrektur-Nachricht, die den Agenten anweist,
    seine vorherige Antwort zu verbessern.
    """
    print("---KORREKTUR-KNOTEN---")
    # Feste Korrektur-Aufforderung für dieses Beispiel
    correction_message = HumanMessage(
        content="KORREKTUR: Die vorherige Antwort war unvollständig. Bitte gib die Temperatur für 'Berlin' in Grad an.",
        name="Validator"
    )
    return {"messages": [correction_message]}

# 4. Logik für die bedingte Weiterleitung (Refaktorisiert)
def should_continue(state: AgentState):
    """
    Bestimmt den nächsten Schritt im Graphen. Diese Funktion ist "pur"
    und verändert den Zustand nicht mehr selbst.
    """
    print("---PRÜFE BEDINGUNG---")
    # Wenn der Zähler > 0 ist, war die letzte Validierung erfolgreich.
    if state.get('validation_passes', 0) > 0:
        print("Bedingung: Validierung war erfolgreich. Ende.")
        return "end"
    else:
        print("Bedingung: Validierung fehlgeschlagen. Gehe zur Korrektur.")
        return "needs_correction"

# 5. Erstellung des Graphen (Refaktorisiert)
def build_graph():
    """Erstellt den LangGraph-Agenten mit einer dedizierten Korrekturschleife."""
    graph = StateGraph(AgentState)

    # Knoten definieren
    graph.add_node("agent", agent_node)
    graph.add_node("validator", validator_node)
    graph.add_node("corrector", correction_node) # Neuer Knoten

    # Einstiegspunkt festlegen
    graph.set_entry_point("agent")

    # Kanten definieren
    graph.add_edge("agent", "validator")
    graph.add_edge("corrector", "agent") # Schleife zurück zum Agenten

    # Bedingte Kante nach der Validierung
    graph.add_conditional_edges(
        "validator",
        should_continue,
        {
            "needs_correction": "corrector", # Route zum neuen Korrektur-Knoten
            "end": END
        }
    )

    # Graph kompilieren
    return graph.compile()

# 6. Hauptausführungsblock
if __name__ == "__main__":
    app = build_graph()

    print("Starte KI-Agenten-Workflow...")
    
    # Testfall 1: Validierung schlägt fehl, Korrekturschleife wird durchlaufen
    print("\n--- Starte Testfall 1: Validierung schlägt fehl & Korrektur ---")
    initial_query_fail = "Wie ist das Wetter für Paris?"
    print(f"Nutzer-Frage: '{initial_query_fail}'")
    
    initial_state_fail = {
        "messages": [HumanMessage(content=initial_query_fail)],
        "validation_passes": 0
    }
    
    final_state_fail = app.invoke(initial_state_fail)

    print("\n--- Workflow Ende (Testfall 1) ---")
    print("Nachrichtenverlauf:")
    for message in final_state_fail['messages']:
        print(f"- {message.name}: {message.content}")
    print(f"Anzahl erfolgreicher Validierungen: {final_state_fail['validation_passes']}")
    
    
    # Testfall 2: Validierung ist direkt erfolgreich
    print("\n\n--- Starte Testfall 2: Validierung erfolgreich im ersten Anlauf ---")
    initial_query_success = "Wie ist das Wetter für Berlin?"
    print(f"Nutzer-Frage: '{initial_query_success}'")

    initial_state_success = {
        "messages": [HumanMessage(content=initial_query_success)],
        "validation_passes": 0
    }
    
    final_state_success = app.invoke(initial_state_success)

    print("\n--- Workflow Ende (Testfall 2) ---")
    print("Nachrichtenverlauf:")
    for message in final_state_success['messages']:
        print(f"- {message.name}: {message.content}")
    print(f"Anzahl erfolgreicher Validierungen: {final_state_success['validation_passes']}")

