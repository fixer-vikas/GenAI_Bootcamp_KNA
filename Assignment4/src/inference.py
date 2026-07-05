from __future__ import annotations

from typing import Dict, Tuple


class HealthcareAssistant:
    def __init__(self) -> None:
        self.rules: Dict[str, Tuple[str, ...]] = {
            "diabetes": (
                "A balanced meal plan, regular movement, and clinician guidance can help manage diabetes. Keep medication as prescribed and monitor symptoms carefully.",
            ),
            "blood pressure": (
                "Reduce salt, stay active, manage stress, and follow your prescribed treatment plan. Regular checkups help track blood pressure safely.",
            ),
            "asthma": (
                "Avoid known triggers, keep an inhaler ready, and seek urgent care if breathing worsens or symptoms become severe.",
            ),
            "burn": (
                "Cool the burn with running water, cover it with a clean dressing, and seek medical care for deeper or larger burns.",
            ),
            "sleep": (
                "Keep a regular bedtime, limit screens before bed, and make the room dark and quiet for better rest.",
            ),
            "vaccin": (
                "Vaccines are generally safe and help prevent serious illness. Mild side effects are common and usually temporary.",
            ),
            "stress": (
                "Try breathing exercises, regular movement, sleep, and support from trusted people or professionals.",
            ),
            "stroke": (
                "Signs such as sudden weakness, facial drooping, trouble speaking, or a severe headache need urgent medical attention.",
            ),
        }

    def generate_answer(self, question: str) -> str:
        text = question.lower()
        for keyword, response in self.rules.items():
            if keyword in text:
                return response
        if any(word in text for word in ["pain", "fever", "cough", "rash"]):
            return "Rest, stay hydrated, and seek medical care if the symptoms are severe, persistent, or worsening."
        return "A healthcare professional can help with that. For urgent symptoms, contact emergency services right away."


def generate_answer(question: str) -> str:
    assistant = HealthcareAssistant()
    return assistant.generate_answer(question)


if __name__ == "__main__":
    sample = "How can I manage diabetes with diet and exercise?"
    print(generate_answer(sample))
