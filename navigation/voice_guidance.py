import pyttsx3


class VoiceGuidance:

    def __init__(self):
        self.engine = pyttsx3.init()

        # Set a reasonable speaking speed
        self.engine.setProperty("rate", 150)

    def speak(self, message):

        print("VOICE:", message)

        self.engine.say(message)
        self.engine.runAndWait()


if __name__ == "__main__":

    voice = VoiceGuidance()

    voice.speak(
        "Navigation started. Move forward."
    )

    voice.speak(
        "Continue straight."
    )

    voice.speak(
        "Turn left."
    )

    voice.speak(
        "You have reached your destination."
    )