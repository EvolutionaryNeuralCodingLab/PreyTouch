export function randomRange (min, max) {
	if (min === 0 && max === 0) {
		return 0
	}
	return Math.floor(Math.random() * (max - min) + min)
}

export const distance = (x1, y1, x2, y2) => Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))

export const plusOrMinus = () => Math.random() < 0.5 ? -1 : 1

export function randBM() {
    let u = 0, v = 0
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

export function shuffle(array) {
  let currentIndex = array.length;
  // While there remain elements to shuffle...
  while (currentIndex != 0) {
    // Pick a remaining element...
    let randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }
}

export function getKeyWithMinFirstArrayValue(obj) {
  if (Object.keys(obj).length === 0) {
    console.warn('Object is empty');
    return
  }
  return Object.keys(obj).reduce((minKey, currentKey) => {
    // Ensure the value is an array with at least one element
    if (!Array.isArray(obj[currentKey]) || obj[currentKey].length === 0) {
      throw new Error(`Value for key "${currentKey}" is not a valid array with at least one element.`);
    }
    return obj[currentKey][0] < obj[minKey][0] ? currentKey : minKey;
  });
}
