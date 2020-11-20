class ActivationFunction {
	constructor(func, dfunc) {
	  this.func = func;
	  this.dfunc = dfunc;
	}
  }
  
  let sigmoid = new ActivationFunction(
	x => 1 / (1 + Math.exp(-x)),
	y => y * (1 - y)
  );
  
  let tanh = new ActivationFunction(
	x => Math.tanh(x),
	y => 1 - (y * y)
  );
  
  
  class NeuralNetwork {
	constructor(a, b, c) {
	  if (a instanceof NeuralNetwork) {
		this.input_nodes = a.input_nodes;
		this.hidden_nodes = a.hidden_nodes;
		this.output_nodes = a.output_nodes;
  
		this.weights_ih = a.weights_ih.copy();
		this.weights_ho = a.weights_ho.copy();
  
		this.bias_h = a.bias_h.copy();
		this.bias_o = a.bias_o.copy();
	  } else {
		this.input_nodes = a;
		this.hidden_nodes = b;
		this.output_nodes = c;
  
		this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
		this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
		this.weights_ih.randomize();
		this.weights_ho.randomize();
  
		this.bias_h = new Matrix(this.hidden_nodes, 1);
		this.bias_o = new Matrix(this.output_nodes, 1);
		this.bias_h.randomize();
		this.bias_o.randomize();
	  }
	  this.setLearningRate();
	  this.setActivationFunction();
	}
  
	predict(input_array) {
	  let inputs = Matrix.fromArray(input_array);
	  let hidden = Matrix.multiply(this.weights_ih, inputs);
	  hidden.add(this.bias_h);
	  hidden.map(this.activation_function.func);

	  let output = Matrix.multiply(this.weights_ho, hidden);
	  output.add(this.bias_o);
	  output.map(this.activation_function.func);

	  return output.toArray();
	}
  
	setLearningRate(learning_rate = 0.1) {
	  this.learning_rate = learning_rate;
	}
  
	setActivationFunction(func = sigmoid) {
	  this.activation_function = func;
	}
  
	train(input_array, target_array) {
	  let inputs = Matrix.fromArray(input_array);
	  let hidden = Matrix.multiply(this.weights_ih, inputs);
	  hidden.add(this.bias_h);
	  hidden.map(this.activation_function.func);
  
	  let outputs = Matrix.multiply(this.weights_ho, hidden);
	  outputs.add(this.bias_o);
	  outputs.map(this.activation_function.func);

	  let targets = Matrix.fromArray(target_array);
  
	  let output_errors = Matrix.subtract(targets, outputs);
	  let gradients = Matrix.map(outputs, this.activation_function.dfunc);
	  gradients.multiply(output_errors);
	  gradients.multiply(this.learning_rate);
  
	  let hidden_T = Matrix.transpose(hidden);
	  let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);
  
	  this.weights_ho.add(weight_ho_deltas);
	  this.bias_o.add(gradients);
  
	  let who_t = Matrix.transpose(this.weights_ho);
	  let hidden_errors = Matrix.multiply(who_t, output_errors);
  	  let hidden_gradient = Matrix.map(hidden, this.activation_function.dfunc);
	  hidden_gradient.multiply(hidden_errors);
	  hidden_gradient.multiply(this.learning_rate);
  
	  let inputs_T = Matrix.transpose(inputs);
	  let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
  
	  this.weights_ih.add(weight_ih_deltas);
	  this.bias_h.add(hidden_gradient);
	}

	copy() {
	  return new NeuralNetwork(this);
	}
  
	mutate(rate) {
	  function mutate(val) {
		if (Math.random() < rate) {
		  return val + randomGaussian(0, 0.1);
		} else {
		  return val;
		}
	  }
	  this.weights_ih.map(mutate);
	  this.weights_ho.map(mutate);
	  this.bias_h.map(mutate);
	  this.bias_o.map(mutate);
	}
}

class Matrix {
	constructor(rows, cols) {
	  this.rows = rows;
	  this.cols = cols;
	  this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
	}
  
	copy() {
	  let m = new Matrix(this.rows, this.cols);
	  for (let i = 0; i < this.rows; i++) {
		for (let j = 0; j < this.cols; j++) {
		  m.data[i][j] = this.data[i][j];
		}
	  }
	  return m;
	}
  
	static fromArray(arr) {
	  return new Matrix(arr.length, 1).map((e, i) => arr[i]);
	}
  
	static subtract(a, b) {
	  if (a.rows !== b.rows || a.cols !== b.cols) {
		console.log('Columns and Rows of A must match Columns and Rows of B.');
		return;
	  }
  
	  return new Matrix(a.rows, a.cols)
		.map((_, i, j) => a.data[i][j] - b.data[i][j]);
	}
  
	toArray() {
	  let arr = [];
	  for (let i = 0; i < this.rows; i++) {
		for (let j = 0; j < this.cols; j++) {
		  arr.push(this.data[i][j]);
		}
	  }
	  return arr;
	}
  
	randomize() {
	  return this.map(e => Math.random() * 2 - 1);
	}
  
	add(n) {
	  if (n instanceof Matrix) {
		if (this.rows !== n.rows || this.cols !== n.cols) {
		  console.log('Columns and Rows of A must match Columns and Rows of B.');
		  return;
		}
		return this.map((e, i, j) => e + n.data[i][j]);
	  } else {
		return this.map(e => e + n);
	  }
	}
  
	static transpose(matrix) {
	  return new Matrix(matrix.cols, matrix.rows)
		.map((_, i, j) => matrix.data[j][i]);
	}
  
	static multiply(a, b) {
	  if (a.cols !== b.rows) {
		console.log('Columns of A must match rows of B.')
		return;
	  }
  
	  return new Matrix(a.rows, b.cols)
		.map((e, i, j) => {
		  let sum = 0;
		  for (let k = 0; k < a.cols; k++) {
			sum += a.data[i][k] * b.data[k][j];
		  }
		  return sum;
		});
	}
  
	multiply(n) {
	  if (n instanceof Matrix) {
		if (this.rows !== n.rows || this.cols !== n.cols) {
		  console.log('Columns and Rows of A must match Columns and Rows of B.');
		  return;
		}
		return this.map((e, i, j) => e * n.data[i][j]);
	  } else {
		return this.map(e => e * n);
	  }
	}
  
	map(func) {
	  for (let i = 0; i < this.rows; i++) {
		for (let j = 0; j < this.cols; j++) {
		  let val = this.data[i][j];
		  this.data[i][j] = func(val, i, j);
		}
	  }
	  return this;
	}
  
	static map(matrix, func) {
	  return new Matrix(matrix.rows, matrix.cols)
		.map((e, i, j) => func(matrix.data[i][j], i, j));
	}
  }