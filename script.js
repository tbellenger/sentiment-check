const loadMetadata = async () => {
  const metadata = await fetch(
    "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json"
  );
  return metadata.json();
};

const loadModel = async () => {
  const url =
    "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json";
  const model = await tf.loadLayersModel(url);
  return model;
};

const padSequences = (sequences, metadata) => {
  return sequences.map((seq) => {
    if (seq.length > metadata.max_len) {
      seq.splice(0, seq.length - metadata.max_len);
    }
    if (seq.length < metadata.max_len) {
      const pad = [];
      for (let i = 0; i < metadata.max_len - seq.length; i++) {
        pad.push(0);
      }
      seq = pad.concat(seq);
    }
    return seq;
  });
};

const predict = (text, model, metadata) => {
  const trimmed = text
    .trim()
    .toLowerCase()
    .replace(/(\.|\,|\!|\?)/g, "")
    .split(" ");
  const sequence = trimmed.map((word) => {
    const wordIndex = metadata.word_index[word];
    if (typeof wordIndex === "undefined") {
      return 2;
    }
    if (wordIndex > 19999) {
      console.log(`wordIndex for ${word} is out of bounds`);
      return 2;
    }
    return wordIndex + metadata.index_from;
  });
  const paddedSequence = padSequences([sequence], metadata);
  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  return score;
};

const getSentiment = (score) => {
  if (score > 0.66) {
    return "positive " + score;
  } else if (score > 0.4) {
    return "neutral " + score;
  } else {
    return "negative " + score;
  }
};

const run = async (text) => {
  const model = await loadModel();
  const metadata = await loadMetadata();
  let sum = 0;
  text.forEach(function (prediction) {
    perc = predict(prediction, model, metadata);
    sum += parseFloat(perc, 10);
  });
  return getSentiment(sum / text.length);
};

window.onload = () => {
  const inputText = document.querySelector("#input-text");
  const button = document.querySelector("#sentiment-button");
  const sentimentEl = document.querySelector("#sentiment");
  inputText.onkeypress = async () => {
    const sentiment = await run([inputText.value]);
    sentimentEl.innerText = sentiment;
  };
};
