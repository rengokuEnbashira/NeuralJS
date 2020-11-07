// Matrix Class
function matrix(n_rows,n_cols){
    this.n_rows = n_rows;
    this.n_cols = n_cols;
    this.data = []
    for(var i = 0;i<n_rows;i++)
	this.data.push(new Array(n_cols));
    this.zeros = function(){
	for(var i = 0;i<this.n_rows;i++)
	    for(var j = 0;j<this.n_cols;j++)
		this.data[i][j] = 0;
    }
    this.ones = function(){
	for(var i = 0;i<this.n_rows;i++)
	    for(var j = 0;j<this.n_cols;j++)
		this.data[i][j] = 1;
    }
    this.rand = function(){
	for(var i = 0;i<this.n_rows;i++)
	    for(var j = 0;j<this.n_cols;j++)
		this.data[i][j] = Math.random();
    }
    this.clone = function(){
	var out = new matrix(this.n_rows,this.n_cols);
	for(var i = 0;i<this.n_rows;i++)
	    for(var j = 0;j<this.n_cols;j++)
		out.data[i][j] = this.data[i][j];
	return out;
    }
    this.add = function(m){
	if(m.n_rows == this.n_rows && m.n_cols == this.n_cols){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] += m.data[i][j];
	    }
	}
	else if(m.n_rows == 1){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] += m.data[0][j];
	    }
	}
    }
    this.diff = function(m){
	if(m.n_rows == this.n_rows && m.n_cols == this.n_cols){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] -= m.data[i][j];
	    }
	}
	else if(m.n_rows == 1){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] -= m.data[0][j];
	    }
	}
    }
    this.div = function(m,axis){
	if(m.n_rows == this.n_rows && m.n_cols == this.n_cols){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] /= m.data[i][j];
	    }
	}
	else if(m.n_rows == 1 && axis == 1){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] /= m.data[0][j];
	    }
	}
	else if(m.n_rows == 1 && axis == 0){
	    for(var i = 0;i<n_rows;i++){
		for(var j = 0;j<n_cols;j++)
		    this.data[i][j] /= m.data[0][i];
	    }
	}
    }
    this.times = function(m){
	for(var i = 0;i<n_rows;i++){
	    for(var j = 0;j<n_cols;j++)
		this.data[i][j] *= m.data[i][j];
	}
    }
    this.scale = function(num){
	for(var i = 0;i<n_rows;i++){
	    for(var j = 0;j<n_cols;j++)
		this.data[i][j] *= num;
	}
    }
    this.dot = function(m){
	var out = new matrix(this.n_rows,m.n_cols);
	var s = 0;
	for(var i = 0;i<this.n_rows;i++){
	    for(var j = 0;j<m.n_cols;j++){
		s = 0;
		for(var k = 0;k<this.n_cols;k++)
		    s += this.data[i][k] * m.data[k][j];
		out.data[i][j] = s;
	    }
	}
	return out;
    }
    this.eval = function(myfun){
	for(var i = 0;i<this.n_rows;i++){
	    for(var j = 0;j<this.n_cols;j++)
		this.data[i][j] = myfun(this.data[i][j]);
	}
    }
    this.transpose = function(){
	var out = new matrix(this.n_cols,this.n_rows);
	for(var i = 0;i<this.n_rows;i++)
	    for(var j = 0;j<this.n_cols;j++)
		out.data[j][i] = this.data[i][j];
	return out;
    }
    this.sum = function(axis){
	var out;
	if(axis == 0){
	    out = new matrix(1,this.n_cols);
	    out.zeros();
	    for(var i = 0;i<this.n_rows;i++)
		for(var j = 0;j<this.n_cols;j++)
		    out.data[0][j] += this.data[i][j];
	}
	else if(axis == 1){
	    out = new matrix(1,this.n_rows);
	    out.zeros();
	    for(var i = 0;i<this.n_rows;i++)
		for(var j = 0;j<this.n_cols;j++)
		    out.data[0][i] += this.data[i][j];
	}
	return out;
    }
    this.from_string = function(str){
	var data = [];
	str = str.replace(/ /g,"");
	var lines = str.split("\n");
	var arr;
	for(var i in lines){
	    arr = lines[i].split(",");
	    var tmp = [];
	    for(var j in arr)
		tmp.push(parseFloat(arr[j]));
	    data.push(tmp);
	}
	this.n_rows = data.length;
	this.n_cols = data[0].length;
	this.data = data;
    }
}


// Activation Functions and their derivatives
function sigmoid(x){
    return 1/(1 + Math.exp(-x));
}

function diff_sigmoid(x){
    return x*(1-x);
}

function tanh(x){
    return (1 - Math.exp(-x))/(1 + Math.exp(-x));
}

function diff_tanh(x){
    return (1 - x**2)/2;
}

function relu(x){
    return x>0?x:0;
}

function diff_relu(x){
    return x>0?1:0;
}

function leaky_relu(x){
    return x>0?x:0.01*x;
}

function diff_leaky_relu(x){
    return x>0?1:0.01;
}

function linear(x){
    return x;
}

function diff_linear(x){
    return 1;
}

dict_act_fun = {"sigmoid":sigmoid,"tanh":tanh,"relu":relu,"leaky_relu":leaky_relu,"linear":linear};
dict_diff_act_fun = {"sigmoid":diff_sigmoid,"tanh":diff_tanh,"relu":diff_relu,"leaky_relu":diff_leaky_relu,"linear":diff_linear};


// Softmax function
function softmax(x){
    var tmp = x.clone();
    tmp.eval(Math.exp);
    tmp.div(tmp.sum(1),0);
    return tmp;
}

// Loss Functions
// input: the output of the network and the target
// output: loss value, gradient of loss 
function mean_square_error(out,target){
    var out_loss = {};
    var tmp = out.clone();
    tmp.diff(target);
    out_loss["grad_loss"] = tmp.clone();
    tmp.times(tmp);
    tmp = tmp.sum(1).sum(0);
    tmp.scale(0.5);
    out_loss["loss"] = tmp.data[0][0];
    return out_loss;
}

function cross_entropy(out,target){
    var tmp = softmax(out);
    var out_loss = {};
    var loss = tmp.clone();
    loss.eval(Math.log);
    loss.times(target);
    loss.scale(-1);
    out_loss["loss"] = loss.sum(1).sum(0).data[0][0];
    tmp.diff(target);
    out_loss["grad_loss"] = tmp;
    return out_loss;
}

dict_loss_func = {"mean_square_error":mean_square_error,"cross_entropy":cross_entropy};


// Block class
// The neural network would be a sequence of these blocks
// Each block performs a linear transformation and applies to this output the specified activation function 
function block(n_inp,n_out,act_fun){
    this.n_inp = n_inp;
    this.n_out = n_out;
    this.weights = new matrix(n_inp,n_out);
    this.weights.rand();
    this.bias = new matrix(1,n_out);
    this.bias.rand();
    this.act_fun = dict_act_fun[act_fun];
    this.diff_act_fun = dict_diff_act_fun[act_fun];
    this.forward = function(x){
	this.stored_inp = x;
	this.stored_out = x.dot(this.weights);
	this.stored_out.add(this.bias);
	this.stored_out.eval(this.act_fun);
	return this.stored_out;
    }
    this.backward = function(err){
	this.delta = this.stored_out.clone();
	this.delta.eval(this.diff_act_fun);
	this.delta.times(err);
	var back_err = this.delta.dot(this.weights.transpose());
	return back_err;
    }
    this.update = function(learning_rate){
	var d_w = this.stored_inp.transpose().dot(this.delta);
	d_w.scale(learning_rate);
	this.weights.diff(d_w);
	var d_b = this.delta.sum(0);
	d_b.scale(learning_rate);
	this.bias.diff(d_b);
    }
}


// Neural Network class
function neural_network(){
    this.blocks = [];
    this.clear = function(){
	this.blocks = [];
    }
    this.add = function(b){
	this.blocks.push(b);
    }
    this.forward = function(x){
	var o = x;
	for(var i in this.blocks)
	    o = this.blocks[i].forward(o);
	return o;
    }
    this.backward = function(x){
	var e = x;
	for(var i = this.blocks.length - 1; i>=0;i--)
	    e = this.blocks[i].backward(e);
	return e;
    }
    this.update = function(learning_rate){
	for(var i in this.blocks)
	    this.blocks[i].update(learning_rate);
    }
    this.train = function(inp_set,target_set,loss_func,maxIt,learning_rate){
	var loss_function = dict_loss_func[loss_func];
	var tmp;
	var out;
	for(var i = 0;i<maxIt;i++){
	    out = this.forward(inp_set);
	    tmp = loss_function(out,target_set);
	    this.backward(tmp["grad_loss"]);
	    this.update(learning_rate);
	}
    }
    this.predict = function(x){
	return this.forward(x);
    }
    this.from_string = function(str){
	var data = JSON.parse(str);
	this.clear();
	for(var layer in data)
	    this.add(new block(parseInt(data[layer]["inp"]),parseInt(data[layer]["out"]),data[layer]["act_fun"]));
    }
}

/*
// Test: Classification of 2d points 

var inp_set = new matrix(20,2);
var out_set = new matrix(20,2);

inp_set.rand();

for(var i = 0;i<20;i++){
    out_set.data[i][0] = inp_set.data[i][0]<inp_set.data[i][1]?1:0;
    out_set.data[i][1] = out_set.data[i][0]?0:1;
}

console.log(out_set.data);

var model = new neural_network();
model.add(new block(2,5,"relu"));
model.add(new block(5,2,"linear"));

model.train(inp_set,out_set,"cross_entropy",300,0.01);

var pred = model.predict(inp_set);
//var out = pred;
var out = softmax(pred);
console.log(out.data);
*/
