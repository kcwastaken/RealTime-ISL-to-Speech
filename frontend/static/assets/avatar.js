export class ISLAvatar {
  constructor(container) {
    this.container = container;
    // Resolves instantly to prevent Three.js crashes before importing your Blender files
    this.ready = Promise.resolve(); 
  }
  
  async playClip(clipName) {
    console.log(`[Avatar System] Playing animation clip: ${clipName}`);
    return true;
  }
}