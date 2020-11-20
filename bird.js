class Bird {
  constructor(network) {
    this.y = height / 2;
    this.x = 64;

    this.gravity = 0.8;
    this.jump = -12;
    this.speed = 0;

    this.score = 0;
    this.fitness = 0;
    if (network) {
      this.network = network.copy();
    } else {
      this.network = new NeuralNetwork(5, 8, 2);
    }
  }

  show() {
    stroke(255);
    fill(255, 100);
    ellipse(this.x, this.y, 32, 32);
  }

  up() {
    this.speed += this.jump;
  }

  mutate() {
    this.network.mutate(0.1);
  }

  think(pipes) {
    let closest = 0;
    let closestD = 1000;
    for (let i = 0; i < pipes.length; i++) {
      let distance = (pipes[i].x + pipes[i].w) - this.x;
      if (distance < closestD && distance > 0) {
        closest = pipes[i];
        closestD = distance;
      }
    }


    let values = [];
    values[0] = this.y / height;
    values[1] = closest.top / height;
    values[2] = closest.bottom / height;
    values[3] = closest.x / width;
    values[4] = this.speed / 10;
    let output = this.network.predict(values);
    if (output[0] > output[1]) {
      this.up();
    }
  }

  offScreen() {
    return (this.y > height || this.y < 0);
  }

  update() {
    this.score++;
    this.speed += this.gravity;
    this.y += this.speed;
  }
}
