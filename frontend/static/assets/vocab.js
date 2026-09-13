const SIGN_VOCAB = [
  { word: "Namaste", emoji: "🙏", video: "Namaste.mp4", avatarClip: "namaste_clip" },
  { word: "Help", emoji: "🤝", video: "Help.mp4", avatarClip: "help_clip" },
  { word: "Doctor", emoji: "🩺", video: "Doctor.mp4", avatarClip: "doctor_clip" },
  { word: "Medicine", emoji: "💊", video: "Medicine.mp4", avatarClip: "medicine_clip" },
  { word: "Water", emoji: "💧", video: "Water.mp4", avatarClip: "water_clip" },
  { word: "Hello", emoji: "👋", video: "Hello.mp4", avatarClip: "hello_clip" }
];

function findVocabEntry(phrase) {
  return SIGN_VOCAB.find(entry => phrase.toLowerCase().includes(entry.word.toLowerCase()));
}