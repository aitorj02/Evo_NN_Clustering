��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.12v2.13.0-17-gf841394b1b78��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
$Adam/v/batch_normalization_5924/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/v/batch_normalization_5924/beta
�
8Adam/v/batch_normalization_5924/beta/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5924/beta*
_output_shapes
:*
dtype0
�
$Adam/m/batch_normalization_5924/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/m/batch_normalization_5924/beta
�
8Adam/m/batch_normalization_5924/beta/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5924/beta*
_output_shapes
:*
dtype0
�
%Adam/v/batch_normalization_5924/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/batch_normalization_5924/gamma
�
9Adam/v/batch_normalization_5924/gamma/Read/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5924/gamma*
_output_shapes
:*
dtype0
�
%Adam/m/batch_normalization_5924/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/batch_normalization_5924/gamma
�
9Adam/m/batch_normalization_5924/gamma/Read/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5924/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_6562/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_6562/bias
}
*Adam/v/dense_6562/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6562/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_6562/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_6562/bias
}
*Adam/m/dense_6562/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6562/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_6562/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_6562/kernel
�
,Adam/v/dense_6562/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6562/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_6562/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_6562/kernel
�
,Adam/m/dense_6562/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6562/kernel*
_output_shapes

:*
dtype0
�
$Adam/v/batch_normalization_5923/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/v/batch_normalization_5923/beta
�
8Adam/v/batch_normalization_5923/beta/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5923/beta*
_output_shapes
:*
dtype0
�
$Adam/m/batch_normalization_5923/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/m/batch_normalization_5923/beta
�
8Adam/m/batch_normalization_5923/beta/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5923/beta*
_output_shapes
:*
dtype0
�
%Adam/v/batch_normalization_5923/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/batch_normalization_5923/gamma
�
9Adam/v/batch_normalization_5923/gamma/Read/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5923/gamma*
_output_shapes
:*
dtype0
�
%Adam/m/batch_normalization_5923/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/batch_normalization_5923/gamma
�
9Adam/m/batch_normalization_5923/gamma/Read/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5923/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_6561/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_6561/bias
}
*Adam/v/dense_6561/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6561/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_6561/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_6561/bias
}
*Adam/m/dense_6561/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6561/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_6561/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_6561/kernel
�
,Adam/v/dense_6561/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6561/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_6561/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_6561/kernel
�
,Adam/m/dense_6561/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6561/kernel*
_output_shapes

:*
dtype0
�
$Adam/v/batch_normalization_5922/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/v/batch_normalization_5922/beta
�
8Adam/v/batch_normalization_5922/beta/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5922/beta*
_output_shapes
:*
dtype0
�
$Adam/m/batch_normalization_5922/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/m/batch_normalization_5922/beta
�
8Adam/m/batch_normalization_5922/beta/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5922/beta*
_output_shapes
:*
dtype0
�
%Adam/v/batch_normalization_5922/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/batch_normalization_5922/gamma
�
9Adam/v/batch_normalization_5922/gamma/Read/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5922/gamma*
_output_shapes
:*
dtype0
�
%Adam/m/batch_normalization_5922/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/batch_normalization_5922/gamma
�
9Adam/m/batch_normalization_5922/gamma/Read/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5922/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_6560/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_6560/bias
}
*Adam/v/dense_6560/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6560/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_6560/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_6560/bias
}
*Adam/m/dense_6560/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6560/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_6560/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_6560/kernel
�
,Adam/v/dense_6560/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6560/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_6560/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_6560/kernel
�
,Adam/m/dense_6560/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6560/kernel*
_output_shapes

:*
dtype0
�
$Adam/v/batch_normalization_5921/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/v/batch_normalization_5921/beta
�
8Adam/v/batch_normalization_5921/beta/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5921/beta*
_output_shapes
:*
dtype0
�
$Adam/m/batch_normalization_5921/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/m/batch_normalization_5921/beta
�
8Adam/m/batch_normalization_5921/beta/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5921/beta*
_output_shapes
:*
dtype0
�
%Adam/v/batch_normalization_5921/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/v/batch_normalization_5921/gamma
�
9Adam/v/batch_normalization_5921/gamma/Read/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5921/gamma*
_output_shapes
:*
dtype0
�
%Adam/m/batch_normalization_5921/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/m/batch_normalization_5921/gamma
�
9Adam/m/batch_normalization_5921/gamma/Read/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5921/gamma*
_output_shapes
:*
dtype0
�
Adam/v/dense_6559/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_6559/bias
}
*Adam/v/dense_6559/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6559/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_6559/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_6559/bias
}
*Adam/m/dense_6559/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6559/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_6559/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/dense_6559/kernel
�
,Adam/v/dense_6559/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6559/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_6559/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/dense_6559/kernel
�
,Adam/m/dense_6559/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6559/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
(batch_normalization_5924/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(batch_normalization_5924/moving_variance
�
<batch_normalization_5924/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_5924/moving_variance*
_output_shapes
:*
dtype0
�
$batch_normalization_5924/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_normalization_5924/moving_mean
�
8batch_normalization_5924/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_5924/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_5924/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_5924/beta
�
1batch_normalization_5924/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5924/beta*
_output_shapes
:*
dtype0
�
batch_normalization_5924/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name batch_normalization_5924/gamma
�
2batch_normalization_5924/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5924/gamma*
_output_shapes
:*
dtype0
v
dense_6562/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_6562/bias
o
#dense_6562/bias/Read/ReadVariableOpReadVariableOpdense_6562/bias*
_output_shapes
:*
dtype0
~
dense_6562/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_6562/kernel
w
%dense_6562/kernel/Read/ReadVariableOpReadVariableOpdense_6562/kernel*
_output_shapes

:*
dtype0
�
(batch_normalization_5923/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(batch_normalization_5923/moving_variance
�
<batch_normalization_5923/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_5923/moving_variance*
_output_shapes
:*
dtype0
�
$batch_normalization_5923/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_normalization_5923/moving_mean
�
8batch_normalization_5923/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_5923/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_5923/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_5923/beta
�
1batch_normalization_5923/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5923/beta*
_output_shapes
:*
dtype0
�
batch_normalization_5923/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name batch_normalization_5923/gamma
�
2batch_normalization_5923/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5923/gamma*
_output_shapes
:*
dtype0
v
dense_6561/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_6561/bias
o
#dense_6561/bias/Read/ReadVariableOpReadVariableOpdense_6561/bias*
_output_shapes
:*
dtype0
~
dense_6561/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_6561/kernel
w
%dense_6561/kernel/Read/ReadVariableOpReadVariableOpdense_6561/kernel*
_output_shapes

:*
dtype0
�
(batch_normalization_5922/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(batch_normalization_5922/moving_variance
�
<batch_normalization_5922/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_5922/moving_variance*
_output_shapes
:*
dtype0
�
$batch_normalization_5922/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_normalization_5922/moving_mean
�
8batch_normalization_5922/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_5922/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_5922/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_5922/beta
�
1batch_normalization_5922/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5922/beta*
_output_shapes
:*
dtype0
�
batch_normalization_5922/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name batch_normalization_5922/gamma
�
2batch_normalization_5922/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5922/gamma*
_output_shapes
:*
dtype0
v
dense_6560/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_6560/bias
o
#dense_6560/bias/Read/ReadVariableOpReadVariableOpdense_6560/bias*
_output_shapes
:*
dtype0
~
dense_6560/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_6560/kernel
w
%dense_6560/kernel/Read/ReadVariableOpReadVariableOpdense_6560/kernel*
_output_shapes

:*
dtype0
�
(batch_normalization_5921/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(batch_normalization_5921/moving_variance
�
<batch_normalization_5921/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_5921/moving_variance*
_output_shapes
:*
dtype0
�
$batch_normalization_5921/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_normalization_5921/moving_mean
�
8batch_normalization_5921/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_5921/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_5921/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_5921/beta
�
1batch_normalization_5921/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5921/beta*
_output_shapes
:*
dtype0
�
batch_normalization_5921/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name batch_normalization_5921/gamma
�
2batch_normalization_5921/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5921/gamma*
_output_shapes
:*
dtype0
v
dense_6559/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_6559/bias
o
#dense_6559/bias/Read/ReadVariableOpReadVariableOpdense_6559/bias*
_output_shapes
:*
dtype0

dense_6559/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namedense_6559/kernel
x
%dense_6559/kernel/Read/ReadVariableOpReadVariableOpdense_6559/kernel*
_output_shapes
:	�*
dtype0

serving_default_input_1051Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1051dense_6559/kerneldense_6559/bias(batch_normalization_5921/moving_variancebatch_normalization_5921/gamma$batch_normalization_5921/moving_meanbatch_normalization_5921/betadense_6560/kerneldense_6560/bias(batch_normalization_5922/moving_variancebatch_normalization_5922/gamma$batch_normalization_5922/moving_meanbatch_normalization_5922/betadense_6561/kerneldense_6561/bias(batch_normalization_5923/moving_variancebatch_normalization_5923/gamma$batch_normalization_5923/moving_meanbatch_normalization_5923/betadense_6562/kerneldense_6562/bias(batch_normalization_5924/moving_variancebatch_normalization_5924/gamma$batch_normalization_5924/moving_meanbatch_normalization_5924/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_331781469

NoOpNoOp
�q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�q
value�qB�q B�q
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance*
�
 0
!1
)2
*3
+4
,5
36
47
<8
=9
>10
?11
F12
G13
O14
P15
Q16
R17
Y18
Z19
b20
c21
d22
e23*
z
 0
!1
)2
*3
34
45
<6
=7
F8
G9
O10
P11
Y12
Z13
b14
c15*
* 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ktrace_0
ltrace_1* 

mtrace_0
ntrace_1* 
* 
�
o
_variables
p_iterations
q_learning_rate
r_index_dict
s
_momentums
t_velocities
u_update_step_xla*

vserving_default* 
* 
* 
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

|trace_0* 

}trace_0* 

 0
!1*

 0
!1*
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_6559/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_6559/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
)0
*1
+2
,3*

)0
*1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
mg
VARIABLE_VALUEbatch_normalization_5921/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_5921/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE$batch_normalization_5921/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE(batch_normalization_5921/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_6560/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_6560/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
<0
=1
>2
?3*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
mg
VARIABLE_VALUEbatch_normalization_5922/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_5922/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE$batch_normalization_5922/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE(batch_normalization_5922/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_6561/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_6561/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
O0
P1
Q2
R3*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
mg
VARIABLE_VALUEbatch_normalization_5923/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_5923/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE$batch_normalization_5923/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE(batch_normalization_5923/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_6562/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_6562/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
b0
c1
d2
e3*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
mg
VARIABLE_VALUEbatch_normalization_5924/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_5924/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE$batch_normalization_5924/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE(batch_normalization_5924/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
<
+0
,1
>2
?3
Q4
R5
d6
e7*
J
0
1
2
3
4
5
6
7
	8

9*

�0
�1*
* 
* 
* 
* 
* 
* 
�
p0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

d0
e1*
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
c]
VARIABLE_VALUEAdam/m/dense_6559/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_6559/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_6559/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_6559/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/batch_normalization_5921/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/batch_normalization_5921/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/batch_normalization_5921/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/batch_normalization_5921/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_6560/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_6560/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_6560/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_6560/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/batch_normalization_5922/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/batch_normalization_5922/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/batch_normalization_5922/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/batch_normalization_5922/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_6561/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_6561/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_6561/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_6561/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/batch_normalization_5923/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/batch_normalization_5923/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/batch_normalization_5923/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/batch_normalization_5923/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_6562/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_6562/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_6562/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_6562/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/m/batch_normalization_5924/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%Adam/v/batch_normalization_5924/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/batch_normalization_5924/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/batch_normalization_5924/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
M
Read/DisableCopyOnReadDisableCopyOnReaddense_6559/kernel"/device:CPU:0
u
Read/ReadVariableOpReadVariableOpdense_6559/kernel"/device:CPU:0*
_output_shapes
:	�*
dtype0
b
IdentityIdentityRead/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
Y

Identity_1IdentityIdentity"/device:CPU:0*
T0*
_output_shapes
:	�
M
Read_1/DisableCopyOnReadDisableCopyOnReaddense_6559/bias"/device:CPU:0
p
Read_1/ReadVariableOpReadVariableOpdense_6559/bias"/device:CPU:0*
_output_shapes
:*
dtype0
a

Identity_2IdentityRead_1/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
V

Identity_3Identity
Identity_2"/device:CPU:0*
T0*
_output_shapes
:
\
Read_2/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5921/gamma"/device:CPU:0

Read_2/ReadVariableOpReadVariableOpbatch_normalization_5921/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
a

Identity_4IdentityRead_2/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
V

Identity_5Identity
Identity_4"/device:CPU:0*
T0*
_output_shapes
:
[
Read_3/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5921/beta"/device:CPU:0
~
Read_3/ReadVariableOpReadVariableOpbatch_normalization_5921/beta"/device:CPU:0*
_output_shapes
:*
dtype0
a

Identity_6IdentityRead_3/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
V

Identity_7Identity
Identity_6"/device:CPU:0*
T0*
_output_shapes
:
b
Read_4/DisableCopyOnReadDisableCopyOnRead$batch_normalization_5921/moving_mean"/device:CPU:0
�
Read_4/ReadVariableOpReadVariableOp$batch_normalization_5921/moving_mean"/device:CPU:0*
_output_shapes
:*
dtype0
a

Identity_8IdentityRead_4/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
V

Identity_9Identity
Identity_8"/device:CPU:0*
T0*
_output_shapes
:
f
Read_5/DisableCopyOnReadDisableCopyOnRead(batch_normalization_5921/moving_variance"/device:CPU:0
�
Read_5/ReadVariableOpReadVariableOp(batch_normalization_5921/moving_variance"/device:CPU:0*
_output_shapes
:*
dtype0
b
Identity_10IdentityRead_5/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_11IdentityIdentity_10"/device:CPU:0*
T0*
_output_shapes
:
O
Read_6/DisableCopyOnReadDisableCopyOnReaddense_6560/kernel"/device:CPU:0
v
Read_6/ReadVariableOpReadVariableOpdense_6560/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
f
Identity_12IdentityRead_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_13IdentityIdentity_12"/device:CPU:0*
T0*
_output_shapes

:
M
Read_7/DisableCopyOnReadDisableCopyOnReaddense_6560/bias"/device:CPU:0
p
Read_7/ReadVariableOpReadVariableOpdense_6560/bias"/device:CPU:0*
_output_shapes
:*
dtype0
b
Identity_14IdentityRead_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_15IdentityIdentity_14"/device:CPU:0*
T0*
_output_shapes
:
\
Read_8/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5922/gamma"/device:CPU:0

Read_8/ReadVariableOpReadVariableOpbatch_normalization_5922/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
b
Identity_16IdentityRead_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_17IdentityIdentity_16"/device:CPU:0*
T0*
_output_shapes
:
[
Read_9/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5922/beta"/device:CPU:0
~
Read_9/ReadVariableOpReadVariableOpbatch_normalization_5922/beta"/device:CPU:0*
_output_shapes
:*
dtype0
b
Identity_18IdentityRead_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_19IdentityIdentity_18"/device:CPU:0*
T0*
_output_shapes
:
c
Read_10/DisableCopyOnReadDisableCopyOnRead$batch_normalization_5922/moving_mean"/device:CPU:0
�
Read_10/ReadVariableOpReadVariableOp$batch_normalization_5922/moving_mean"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_20IdentityRead_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_21IdentityIdentity_20"/device:CPU:0*
T0*
_output_shapes
:
g
Read_11/DisableCopyOnReadDisableCopyOnRead(batch_normalization_5922/moving_variance"/device:CPU:0
�
Read_11/ReadVariableOpReadVariableOp(batch_normalization_5922/moving_variance"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_22IdentityRead_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_23IdentityIdentity_22"/device:CPU:0*
T0*
_output_shapes
:
P
Read_12/DisableCopyOnReadDisableCopyOnReaddense_6561/kernel"/device:CPU:0
w
Read_12/ReadVariableOpReadVariableOpdense_6561/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
g
Identity_24IdentityRead_12/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_25IdentityIdentity_24"/device:CPU:0*
T0*
_output_shapes

:
N
Read_13/DisableCopyOnReadDisableCopyOnReaddense_6561/bias"/device:CPU:0
q
Read_13/ReadVariableOpReadVariableOpdense_6561/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_26IdentityRead_13/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_27IdentityIdentity_26"/device:CPU:0*
T0*
_output_shapes
:
]
Read_14/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5923/gamma"/device:CPU:0
�
Read_14/ReadVariableOpReadVariableOpbatch_normalization_5923/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_28IdentityRead_14/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_29IdentityIdentity_28"/device:CPU:0*
T0*
_output_shapes
:
\
Read_15/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5923/beta"/device:CPU:0

Read_15/ReadVariableOpReadVariableOpbatch_normalization_5923/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_30IdentityRead_15/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_31IdentityIdentity_30"/device:CPU:0*
T0*
_output_shapes
:
c
Read_16/DisableCopyOnReadDisableCopyOnRead$batch_normalization_5923/moving_mean"/device:CPU:0
�
Read_16/ReadVariableOpReadVariableOp$batch_normalization_5923/moving_mean"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_32IdentityRead_16/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_33IdentityIdentity_32"/device:CPU:0*
T0*
_output_shapes
:
g
Read_17/DisableCopyOnReadDisableCopyOnRead(batch_normalization_5923/moving_variance"/device:CPU:0
�
Read_17/ReadVariableOpReadVariableOp(batch_normalization_5923/moving_variance"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_34IdentityRead_17/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_35IdentityIdentity_34"/device:CPU:0*
T0*
_output_shapes
:
P
Read_18/DisableCopyOnReadDisableCopyOnReaddense_6562/kernel"/device:CPU:0
w
Read_18/ReadVariableOpReadVariableOpdense_6562/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
g
Identity_36IdentityRead_18/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_37IdentityIdentity_36"/device:CPU:0*
T0*
_output_shapes

:
N
Read_19/DisableCopyOnReadDisableCopyOnReaddense_6562/bias"/device:CPU:0
q
Read_19/ReadVariableOpReadVariableOpdense_6562/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_38IdentityRead_19/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_39IdentityIdentity_38"/device:CPU:0*
T0*
_output_shapes
:
]
Read_20/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5924/gamma"/device:CPU:0
�
Read_20/ReadVariableOpReadVariableOpbatch_normalization_5924/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_40IdentityRead_20/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_41IdentityIdentity_40"/device:CPU:0*
T0*
_output_shapes
:
\
Read_21/DisableCopyOnReadDisableCopyOnReadbatch_normalization_5924/beta"/device:CPU:0

Read_21/ReadVariableOpReadVariableOpbatch_normalization_5924/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_42IdentityRead_21/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_43IdentityIdentity_42"/device:CPU:0*
T0*
_output_shapes
:
c
Read_22/DisableCopyOnReadDisableCopyOnRead$batch_normalization_5924/moving_mean"/device:CPU:0
�
Read_22/ReadVariableOpReadVariableOp$batch_normalization_5924/moving_mean"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_44IdentityRead_22/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_45IdentityIdentity_44"/device:CPU:0*
T0*
_output_shapes
:
g
Read_23/DisableCopyOnReadDisableCopyOnRead(batch_normalization_5924/moving_variance"/device:CPU:0
�
Read_23/ReadVariableOpReadVariableOp(batch_normalization_5924/moving_variance"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_46IdentityRead_23/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_47IdentityIdentity_46"/device:CPU:0*
T0*
_output_shapes
:
H
Read_24/DisableCopyOnReadDisableCopyOnRead	iteration"/device:CPU:0
g
Read_24/ReadVariableOpReadVariableOp	iteration"/device:CPU:0*
_output_shapes
: *
dtype0	
_
Identity_48IdentityRead_24/ReadVariableOp"/device:CPU:0*
T0	*
_output_shapes
: 
T
Identity_49IdentityIdentity_48"/device:CPU:0*
T0	*
_output_shapes
: 
L
Read_25/DisableCopyOnReadDisableCopyOnReadlearning_rate"/device:CPU:0
k
Read_25/ReadVariableOpReadVariableOplearning_rate"/device:CPU:0*
_output_shapes
: *
dtype0
_
Identity_50IdentityRead_25/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
T
Identity_51IdentityIdentity_50"/device:CPU:0*
T0*
_output_shapes
: 
W
Read_26/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6559/kernel"/device:CPU:0

Read_26/ReadVariableOpReadVariableOpAdam/m/dense_6559/kernel"/device:CPU:0*
_output_shapes
:	�*
dtype0
h
Identity_52IdentityRead_26/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
]
Identity_53IdentityIdentity_52"/device:CPU:0*
T0*
_output_shapes
:	�
W
Read_27/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6559/kernel"/device:CPU:0

Read_27/ReadVariableOpReadVariableOpAdam/v/dense_6559/kernel"/device:CPU:0*
_output_shapes
:	�*
dtype0
h
Identity_54IdentityRead_27/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
]
Identity_55IdentityIdentity_54"/device:CPU:0*
T0*
_output_shapes
:	�
U
Read_28/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6559/bias"/device:CPU:0
x
Read_28/ReadVariableOpReadVariableOpAdam/m/dense_6559/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_56IdentityRead_28/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_57IdentityIdentity_56"/device:CPU:0*
T0*
_output_shapes
:
U
Read_29/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6559/bias"/device:CPU:0
x
Read_29/ReadVariableOpReadVariableOpAdam/v/dense_6559/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_58IdentityRead_29/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_59IdentityIdentity_58"/device:CPU:0*
T0*
_output_shapes
:
d
Read_30/DisableCopyOnReadDisableCopyOnRead%Adam/m/batch_normalization_5921/gamma"/device:CPU:0
�
Read_30/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5921/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_60IdentityRead_30/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_61IdentityIdentity_60"/device:CPU:0*
T0*
_output_shapes
:
d
Read_31/DisableCopyOnReadDisableCopyOnRead%Adam/v/batch_normalization_5921/gamma"/device:CPU:0
�
Read_31/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5921/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_62IdentityRead_31/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_63IdentityIdentity_62"/device:CPU:0*
T0*
_output_shapes
:
c
Read_32/DisableCopyOnReadDisableCopyOnRead$Adam/m/batch_normalization_5921/beta"/device:CPU:0
�
Read_32/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5921/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_64IdentityRead_32/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_65IdentityIdentity_64"/device:CPU:0*
T0*
_output_shapes
:
c
Read_33/DisableCopyOnReadDisableCopyOnRead$Adam/v/batch_normalization_5921/beta"/device:CPU:0
�
Read_33/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5921/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_66IdentityRead_33/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_67IdentityIdentity_66"/device:CPU:0*
T0*
_output_shapes
:
W
Read_34/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6560/kernel"/device:CPU:0
~
Read_34/ReadVariableOpReadVariableOpAdam/m/dense_6560/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
g
Identity_68IdentityRead_34/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_69IdentityIdentity_68"/device:CPU:0*
T0*
_output_shapes

:
W
Read_35/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6560/kernel"/device:CPU:0
~
Read_35/ReadVariableOpReadVariableOpAdam/v/dense_6560/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
g
Identity_70IdentityRead_35/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_71IdentityIdentity_70"/device:CPU:0*
T0*
_output_shapes

:
U
Read_36/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6560/bias"/device:CPU:0
x
Read_36/ReadVariableOpReadVariableOpAdam/m/dense_6560/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_72IdentityRead_36/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_73IdentityIdentity_72"/device:CPU:0*
T0*
_output_shapes
:
U
Read_37/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6560/bias"/device:CPU:0
x
Read_37/ReadVariableOpReadVariableOpAdam/v/dense_6560/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_74IdentityRead_37/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_75IdentityIdentity_74"/device:CPU:0*
T0*
_output_shapes
:
d
Read_38/DisableCopyOnReadDisableCopyOnRead%Adam/m/batch_normalization_5922/gamma"/device:CPU:0
�
Read_38/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5922/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_76IdentityRead_38/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_77IdentityIdentity_76"/device:CPU:0*
T0*
_output_shapes
:
d
Read_39/DisableCopyOnReadDisableCopyOnRead%Adam/v/batch_normalization_5922/gamma"/device:CPU:0
�
Read_39/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5922/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_78IdentityRead_39/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_79IdentityIdentity_78"/device:CPU:0*
T0*
_output_shapes
:
c
Read_40/DisableCopyOnReadDisableCopyOnRead$Adam/m/batch_normalization_5922/beta"/device:CPU:0
�
Read_40/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5922/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_80IdentityRead_40/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_81IdentityIdentity_80"/device:CPU:0*
T0*
_output_shapes
:
c
Read_41/DisableCopyOnReadDisableCopyOnRead$Adam/v/batch_normalization_5922/beta"/device:CPU:0
�
Read_41/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5922/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_82IdentityRead_41/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_83IdentityIdentity_82"/device:CPU:0*
T0*
_output_shapes
:
W
Read_42/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6561/kernel"/device:CPU:0
~
Read_42/ReadVariableOpReadVariableOpAdam/m/dense_6561/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
g
Identity_84IdentityRead_42/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_85IdentityIdentity_84"/device:CPU:0*
T0*
_output_shapes

:
W
Read_43/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6561/kernel"/device:CPU:0
~
Read_43/ReadVariableOpReadVariableOpAdam/v/dense_6561/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
g
Identity_86IdentityRead_43/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
\
Identity_87IdentityIdentity_86"/device:CPU:0*
T0*
_output_shapes

:
U
Read_44/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6561/bias"/device:CPU:0
x
Read_44/ReadVariableOpReadVariableOpAdam/m/dense_6561/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_88IdentityRead_44/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_89IdentityIdentity_88"/device:CPU:0*
T0*
_output_shapes
:
U
Read_45/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6561/bias"/device:CPU:0
x
Read_45/ReadVariableOpReadVariableOpAdam/v/dense_6561/bias"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_90IdentityRead_45/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_91IdentityIdentity_90"/device:CPU:0*
T0*
_output_shapes
:
d
Read_46/DisableCopyOnReadDisableCopyOnRead%Adam/m/batch_normalization_5923/gamma"/device:CPU:0
�
Read_46/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5923/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_92IdentityRead_46/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_93IdentityIdentity_92"/device:CPU:0*
T0*
_output_shapes
:
d
Read_47/DisableCopyOnReadDisableCopyOnRead%Adam/v/batch_normalization_5923/gamma"/device:CPU:0
�
Read_47/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5923/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_94IdentityRead_47/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_95IdentityIdentity_94"/device:CPU:0*
T0*
_output_shapes
:
c
Read_48/DisableCopyOnReadDisableCopyOnRead$Adam/m/batch_normalization_5923/beta"/device:CPU:0
�
Read_48/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5923/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_96IdentityRead_48/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_97IdentityIdentity_96"/device:CPU:0*
T0*
_output_shapes
:
c
Read_49/DisableCopyOnReadDisableCopyOnRead$Adam/v/batch_normalization_5923/beta"/device:CPU:0
�
Read_49/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5923/beta"/device:CPU:0*
_output_shapes
:*
dtype0
c
Identity_98IdentityRead_49/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_99IdentityIdentity_98"/device:CPU:0*
T0*
_output_shapes
:
W
Read_50/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6562/kernel"/device:CPU:0
~
Read_50/ReadVariableOpReadVariableOpAdam/m/dense_6562/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
h
Identity_100IdentityRead_50/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
^
Identity_101IdentityIdentity_100"/device:CPU:0*
T0*
_output_shapes

:
W
Read_51/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6562/kernel"/device:CPU:0
~
Read_51/ReadVariableOpReadVariableOpAdam/v/dense_6562/kernel"/device:CPU:0*
_output_shapes

:*
dtype0
h
Identity_102IdentityRead_51/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
^
Identity_103IdentityIdentity_102"/device:CPU:0*
T0*
_output_shapes

:
U
Read_52/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_6562/bias"/device:CPU:0
x
Read_52/ReadVariableOpReadVariableOpAdam/m/dense_6562/bias"/device:CPU:0*
_output_shapes
:*
dtype0
d
Identity_104IdentityRead_52/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
Z
Identity_105IdentityIdentity_104"/device:CPU:0*
T0*
_output_shapes
:
U
Read_53/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_6562/bias"/device:CPU:0
x
Read_53/ReadVariableOpReadVariableOpAdam/v/dense_6562/bias"/device:CPU:0*
_output_shapes
:*
dtype0
d
Identity_106IdentityRead_53/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
Z
Identity_107IdentityIdentity_106"/device:CPU:0*
T0*
_output_shapes
:
d
Read_54/DisableCopyOnReadDisableCopyOnRead%Adam/m/batch_normalization_5924/gamma"/device:CPU:0
�
Read_54/ReadVariableOpReadVariableOp%Adam/m/batch_normalization_5924/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
d
Identity_108IdentityRead_54/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
Z
Identity_109IdentityIdentity_108"/device:CPU:0*
T0*
_output_shapes
:
d
Read_55/DisableCopyOnReadDisableCopyOnRead%Adam/v/batch_normalization_5924/gamma"/device:CPU:0
�
Read_55/ReadVariableOpReadVariableOp%Adam/v/batch_normalization_5924/gamma"/device:CPU:0*
_output_shapes
:*
dtype0
d
Identity_110IdentityRead_55/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
Z
Identity_111IdentityIdentity_110"/device:CPU:0*
T0*
_output_shapes
:
c
Read_56/DisableCopyOnReadDisableCopyOnRead$Adam/m/batch_normalization_5924/beta"/device:CPU:0
�
Read_56/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_5924/beta"/device:CPU:0*
_output_shapes
:*
dtype0
d
Identity_112IdentityRead_56/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
Z
Identity_113IdentityIdentity_112"/device:CPU:0*
T0*
_output_shapes
:
c
Read_57/DisableCopyOnReadDisableCopyOnRead$Adam/v/batch_normalization_5924/beta"/device:CPU:0
�
Read_57/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_5924/beta"/device:CPU:0*
_output_shapes
:*
dtype0
d
Identity_114IdentityRead_57/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
Z
Identity_115IdentityIdentity_114"/device:CPU:0*
T0*
_output_shapes
:
F
Read_58/DisableCopyOnReadDisableCopyOnReadtotal_1"/device:CPU:0
e
Read_58/ReadVariableOpReadVariableOptotal_1"/device:CPU:0*
_output_shapes
: *
dtype0
`
Identity_116IdentityRead_58/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
V
Identity_117IdentityIdentity_116"/device:CPU:0*
T0*
_output_shapes
: 
F
Read_59/DisableCopyOnReadDisableCopyOnReadcount_1"/device:CPU:0
e
Read_59/ReadVariableOpReadVariableOpcount_1"/device:CPU:0*
_output_shapes
: *
dtype0
`
Identity_118IdentityRead_59/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
V
Identity_119IdentityIdentity_118"/device:CPU:0*
T0*
_output_shapes
: 
D
Read_60/DisableCopyOnReadDisableCopyOnReadtotal"/device:CPU:0
c
Read_60/ReadVariableOpReadVariableOptotal"/device:CPU:0*
_output_shapes
: *
dtype0
`
Identity_120IdentityRead_60/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
V
Identity_121IdentityIdentity_120"/device:CPU:0*
T0*
_output_shapes
: 
D
Read_61/DisableCopyOnReadDisableCopyOnReadcount"/device:CPU:0
c
Read_61/ReadVariableOpReadVariableOpcount"/device:CPU:0*
_output_shapes
: *
dtype0
`
Identity_122IdentityRead_61/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
V
Identity_123IdentityIdentity_122"/device:CPU:0*
T0*
_output_shapes
: 
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�
value�B�?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�
value�B�?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices
Identity_1
Identity_3
Identity_5
Identity_7
Identity_9Identity_11Identity_13Identity_15Identity_17Identity_19Identity_21Identity_23Identity_25Identity_27Identity_29Identity_31Identity_33Identity_35Identity_37Identity_39Identity_41Identity_43Identity_45Identity_47Identity_49Identity_51Identity_53Identity_55Identity_57Identity_59Identity_61Identity_63Identity_65Identity_67Identity_69Identity_71Identity_73Identity_75Identity_77Identity_79Identity_81Identity_83Identity_85Identity_87Identity_89Identity_91Identity_93Identity_95Identity_97Identity_99Identity_101Identity_103Identity_105Identity_107Identity_109Identity_111Identity_113Identity_115Identity_117Identity_119Identity_121Identity_123Const"/device:CPU:0*&
 _has_manual_control_dependencies(*M
dtypesC
A2?	
�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
�
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0*&
 _has_manual_control_dependencies(
m
Identity_124Identitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�
value�B�?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�
value�B�?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	
U
Identity_125Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOpAssignVariableOpdense_6559/kernelIdentity_125"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_126IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_1AssignVariableOpdense_6559/biasIdentity_126"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_127IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_2AssignVariableOpbatch_normalization_5921/gammaIdentity_127"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_128IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_3AssignVariableOpbatch_normalization_5921/betaIdentity_128"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_129IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_4AssignVariableOp$batch_normalization_5921/moving_meanIdentity_129"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_130IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_5AssignVariableOp(batch_normalization_5921/moving_varianceIdentity_130"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_131IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_6AssignVariableOpdense_6560/kernelIdentity_131"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_132IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_7AssignVariableOpdense_6560/biasIdentity_132"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_133IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_8AssignVariableOpbatch_normalization_5922/gammaIdentity_133"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_134IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_9AssignVariableOpbatch_normalization_5922/betaIdentity_134"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_135IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_10AssignVariableOp$batch_normalization_5922/moving_meanIdentity_135"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_136IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_11AssignVariableOp(batch_normalization_5922/moving_varianceIdentity_136"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_137IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_12AssignVariableOpdense_6561/kernelIdentity_137"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_138IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_13AssignVariableOpdense_6561/biasIdentity_138"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_139IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_14AssignVariableOpbatch_normalization_5923/gammaIdentity_139"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_140IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_15AssignVariableOpbatch_normalization_5923/betaIdentity_140"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_141IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_16AssignVariableOp$batch_normalization_5923/moving_meanIdentity_141"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_142IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_17AssignVariableOp(batch_normalization_5923/moving_varianceIdentity_142"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_143IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_18AssignVariableOpdense_6562/kernelIdentity_143"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_144IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_19AssignVariableOpdense_6562/biasIdentity_144"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_145IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_20AssignVariableOpbatch_normalization_5924/gammaIdentity_145"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_146IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_21AssignVariableOpbatch_normalization_5924/betaIdentity_146"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_147IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_22AssignVariableOp$batch_normalization_5924/moving_meanIdentity_147"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_148IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_23AssignVariableOp(batch_normalization_5924/moving_varianceIdentity_148"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_149IdentityRestoreV2:24"/device:CPU:0*
T0	*
_output_shapes
:
�
AssignVariableOp_24AssignVariableOp	iterationIdentity_149"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0	
X
Identity_150IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_25AssignVariableOplearning_rateIdentity_150"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_151IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_26AssignVariableOpAdam/m/dense_6559/kernelIdentity_151"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_152IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_27AssignVariableOpAdam/v/dense_6559/kernelIdentity_152"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_153IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_28AssignVariableOpAdam/m/dense_6559/biasIdentity_153"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_154IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_29AssignVariableOpAdam/v/dense_6559/biasIdentity_154"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_155IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_30AssignVariableOp%Adam/m/batch_normalization_5921/gammaIdentity_155"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_156IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_31AssignVariableOp%Adam/v/batch_normalization_5921/gammaIdentity_156"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_157IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_32AssignVariableOp$Adam/m/batch_normalization_5921/betaIdentity_157"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_158IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_33AssignVariableOp$Adam/v/batch_normalization_5921/betaIdentity_158"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_159IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_34AssignVariableOpAdam/m/dense_6560/kernelIdentity_159"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_160IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_35AssignVariableOpAdam/v/dense_6560/kernelIdentity_160"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_161IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_36AssignVariableOpAdam/m/dense_6560/biasIdentity_161"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_162IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_37AssignVariableOpAdam/v/dense_6560/biasIdentity_162"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_163IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_38AssignVariableOp%Adam/m/batch_normalization_5922/gammaIdentity_163"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_164IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_39AssignVariableOp%Adam/v/batch_normalization_5922/gammaIdentity_164"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_165IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_40AssignVariableOp$Adam/m/batch_normalization_5922/betaIdentity_165"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_166IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_41AssignVariableOp$Adam/v/batch_normalization_5922/betaIdentity_166"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_167IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_42AssignVariableOpAdam/m/dense_6561/kernelIdentity_167"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_168IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_43AssignVariableOpAdam/v/dense_6561/kernelIdentity_168"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_169IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_44AssignVariableOpAdam/m/dense_6561/biasIdentity_169"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_170IdentityRestoreV2:45"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_45AssignVariableOpAdam/v/dense_6561/biasIdentity_170"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_171IdentityRestoreV2:46"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_46AssignVariableOp%Adam/m/batch_normalization_5923/gammaIdentity_171"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_172IdentityRestoreV2:47"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_47AssignVariableOp%Adam/v/batch_normalization_5923/gammaIdentity_172"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_173IdentityRestoreV2:48"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_48AssignVariableOp$Adam/m/batch_normalization_5923/betaIdentity_173"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_174IdentityRestoreV2:49"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_49AssignVariableOp$Adam/v/batch_normalization_5923/betaIdentity_174"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_175IdentityRestoreV2:50"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_50AssignVariableOpAdam/m/dense_6562/kernelIdentity_175"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_176IdentityRestoreV2:51"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_51AssignVariableOpAdam/v/dense_6562/kernelIdentity_176"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_177IdentityRestoreV2:52"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_52AssignVariableOpAdam/m/dense_6562/biasIdentity_177"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_178IdentityRestoreV2:53"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_53AssignVariableOpAdam/v/dense_6562/biasIdentity_178"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_179IdentityRestoreV2:54"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_54AssignVariableOp%Adam/m/batch_normalization_5924/gammaIdentity_179"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_180IdentityRestoreV2:55"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_55AssignVariableOp%Adam/v/batch_normalization_5924/gammaIdentity_180"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_181IdentityRestoreV2:56"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_56AssignVariableOp$Adam/m/batch_normalization_5924/betaIdentity_181"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_182IdentityRestoreV2:57"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_57AssignVariableOp$Adam/v/batch_normalization_5924/betaIdentity_182"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_183IdentityRestoreV2:58"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_58AssignVariableOptotal_1Identity_183"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_184IdentityRestoreV2:59"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_59AssignVariableOpcount_1Identity_184"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_185IdentityRestoreV2:60"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_60AssignVariableOptotalIdentity_185"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
X
Identity_186IdentityRestoreV2:61"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_61AssignVariableOpcountIdentity_186"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
E
NoOp_1NoOp"/device:CPU:0*&
 _has_manual_control_dependencies(
�
Identity_187Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: �
�

�
I__inference_dense_6562_layer_call_and_return_conditional_losses_331781891

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_dense_6561_layer_call_fn_331781750

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781589

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
<__inference_batch_normalization_5924_layer_call_fn_331781945

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_331781481

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
I__inference_model_1050_layer_call_and_return_conditional_losses_331780968

input_1051<
)dense_6559_matmul_readvariableop_resource:	�8
*dense_6559_biasadd_readvariableop_resource:N
@batch_normalization_5921_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5921_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5921_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5921_batchnorm_readvariableop_resource:;
)dense_6560_matmul_readvariableop_resource:8
*dense_6560_biasadd_readvariableop_resource:N
@batch_normalization_5922_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5922_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5922_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5922_batchnorm_readvariableop_resource:;
)dense_6561_matmul_readvariableop_resource:8
*dense_6561_biasadd_readvariableop_resource:N
@batch_normalization_5923_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5923_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5923_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5923_batchnorm_readvariableop_resource:;
)dense_6562_matmul_readvariableop_resource:8
*dense_6562_biasadd_readvariableop_resource:N
@batch_normalization_5924_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5924_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5924_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5924_batchnorm_readvariableop_resource:
identity��(batch_normalization_5921/AssignMovingAvg�7batch_normalization_5921/AssignMovingAvg/ReadVariableOp�*batch_normalization_5921/AssignMovingAvg_1�9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5921/batchnorm/ReadVariableOp�5batch_normalization_5921/batchnorm/mul/ReadVariableOp�(batch_normalization_5922/AssignMovingAvg�7batch_normalization_5922/AssignMovingAvg/ReadVariableOp�*batch_normalization_5922/AssignMovingAvg_1�9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5922/batchnorm/ReadVariableOp�5batch_normalization_5922/batchnorm/mul/ReadVariableOp�(batch_normalization_5923/AssignMovingAvg�7batch_normalization_5923/AssignMovingAvg/ReadVariableOp�*batch_normalization_5923/AssignMovingAvg_1�9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5923/batchnorm/ReadVariableOp�5batch_normalization_5923/batchnorm/mul/ReadVariableOp�(batch_normalization_5924/AssignMovingAvg�7batch_normalization_5924/AssignMovingAvg/ReadVariableOp�*batch_normalization_5924/AssignMovingAvg_1�9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5924/batchnorm/ReadVariableOp�5batch_normalization_5924/batchnorm/mul/ReadVariableOp�!dense_6559/BiasAdd/ReadVariableOp� dense_6559/MatMul/ReadVariableOp�!dense_6560/BiasAdd/ReadVariableOp� dense_6560/MatMul/ReadVariableOp�!dense_6561/BiasAdd/ReadVariableOp� dense_6561/MatMul/ReadVariableOp�!dense_6562/BiasAdd/ReadVariableOp� dense_6562/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  q
flatten/ReshapeReshape
input_1051flatten/Const:output:0*
T0*(
_output_shapes
:�����������
 dense_6559/MatMul/ReadVariableOpReadVariableOp)dense_6559_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_6559/MatMulMatMulflatten/Reshape:output:0(dense_6559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6559/BiasAdd/ReadVariableOpReadVariableOp*dense_6559_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6559/BiasAddBiasAdddense_6559/MatMul:product:0)dense_6559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7batch_normalization_5921/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5921/moments/meanMeandense_6559/BiasAdd:output:0@batch_normalization_5921/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5921/moments/StopGradientStopGradient.batch_normalization_5921/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5921/moments/SquaredDifferenceSquaredDifferencedense_6559/BiasAdd:output:06batch_normalization_5921/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5921/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5921/moments/varianceMean6batch_normalization_5921/moments/SquaredDifference:z:0Dbatch_normalization_5921/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5921/moments/SqueezeSqueeze.batch_normalization_5921/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5921/moments/Squeeze_1Squeeze2batch_normalization_5921/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5921/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5921/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5921_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5921/AssignMovingAvg/subSub?batch_normalization_5921/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5921/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5921/AssignMovingAvg/mulMul0batch_normalization_5921/AssignMovingAvg/sub:z:07batch_normalization_5921/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5921/AssignMovingAvgAssignSubVariableOp@batch_normalization_5921_assignmovingavg_readvariableop_resource0batch_normalization_5921/AssignMovingAvg/mul:z:08^batch_normalization_5921/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5921/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5921_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5921/AssignMovingAvg_1/subSubAbatch_normalization_5921/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5921/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5921/AssignMovingAvg_1/mulMul2batch_normalization_5921/AssignMovingAvg_1/sub:z:09batch_normalization_5921/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5921/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5921_assignmovingavg_1_readvariableop_resource2batch_normalization_5921/AssignMovingAvg_1/mul:z:0:^batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5921/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5921/batchnorm/addAddV23batch_normalization_5921/moments/Squeeze_1:output:05batch_normalization_5921/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/RsqrtRsqrt*batch_normalization_5921/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5921/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5921_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5921/batchnorm/mul/mulMul,batch_normalization_5921/batchnorm/Rsqrt:y:0=batch_normalization_5921/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/mul_1Muldense_6559/BiasAdd:output:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5921/batchnorm/mul_2Mul1batch_normalization_5921/moments/Squeeze:output:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5921/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5921_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5921/batchnorm/subSub9batch_normalization_5921/batchnorm/ReadVariableOp:value:0,batch_normalization_5921/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/add_1AddV2,batch_normalization_5921/batchnorm/mul_1:z:0*batch_normalization_5921/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6560/MatMul/ReadVariableOpReadVariableOp)dense_6560_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6560/MatMulMatMul,batch_normalization_5921/batchnorm/add_1:z:0(dense_6560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6560/BiasAdd/ReadVariableOpReadVariableOp*dense_6560_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6560/BiasAddBiasAdddense_6560/MatMul:product:0)dense_6560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6560/TanhTanhdense_6560/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7batch_normalization_5922/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5922/moments/meanMeandense_6560/Tanh:y:0@batch_normalization_5922/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5922/moments/StopGradientStopGradient.batch_normalization_5922/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5922/moments/SquaredDifferenceSquaredDifferencedense_6560/Tanh:y:06batch_normalization_5922/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5922/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5922/moments/varianceMean6batch_normalization_5922/moments/SquaredDifference:z:0Dbatch_normalization_5922/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5922/moments/SqueezeSqueeze.batch_normalization_5922/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5922/moments/Squeeze_1Squeeze2batch_normalization_5922/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5922/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5922/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5922_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5922/AssignMovingAvg/subSub?batch_normalization_5922/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5922/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5922/AssignMovingAvg/mulMul0batch_normalization_5922/AssignMovingAvg/sub:z:07batch_normalization_5922/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5922/AssignMovingAvgAssignSubVariableOp@batch_normalization_5922_assignmovingavg_readvariableop_resource0batch_normalization_5922/AssignMovingAvg/mul:z:08^batch_normalization_5922/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5922/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5922_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5922/AssignMovingAvg_1/subSubAbatch_normalization_5922/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5922/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5922/AssignMovingAvg_1/mulMul2batch_normalization_5922/AssignMovingAvg_1/sub:z:09batch_normalization_5922/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5922/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5922_assignmovingavg_1_readvariableop_resource2batch_normalization_5922/AssignMovingAvg_1/mul:z:0:^batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5922/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5922/batchnorm/addAddV23batch_normalization_5922/moments/Squeeze_1:output:05batch_normalization_5922/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/RsqrtRsqrt*batch_normalization_5922/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5922/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5922_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5922/batchnorm/mul/mulMul,batch_normalization_5922/batchnorm/Rsqrt:y:0=batch_normalization_5922/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/mul_1Muldense_6560/Tanh:y:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5922/batchnorm/mul_2Mul1batch_normalization_5922/moments/Squeeze:output:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5922/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5922_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5922/batchnorm/subSub9batch_normalization_5922/batchnorm/ReadVariableOp:value:0,batch_normalization_5922/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/add_1AddV2,batch_normalization_5922/batchnorm/mul_1:z:0*batch_normalization_5922/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6561/MatMul/ReadVariableOpReadVariableOp)dense_6561_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6561/MatMulMatMul,batch_normalization_5922/batchnorm/add_1:z:0(dense_6561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6561/BiasAdd/ReadVariableOpReadVariableOp*dense_6561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6561/BiasAddBiasAdddense_6561/MatMul:product:0)dense_6561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6561/TanhTanhdense_6561/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7batch_normalization_5923/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5923/moments/meanMeandense_6561/Tanh:y:0@batch_normalization_5923/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5923/moments/StopGradientStopGradient.batch_normalization_5923/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5923/moments/SquaredDifferenceSquaredDifferencedense_6561/Tanh:y:06batch_normalization_5923/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5923/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5923/moments/varianceMean6batch_normalization_5923/moments/SquaredDifference:z:0Dbatch_normalization_5923/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5923/moments/SqueezeSqueeze.batch_normalization_5923/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5923/moments/Squeeze_1Squeeze2batch_normalization_5923/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5923/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5923/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5923_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5923/AssignMovingAvg/subSub?batch_normalization_5923/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5923/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5923/AssignMovingAvg/mulMul0batch_normalization_5923/AssignMovingAvg/sub:z:07batch_normalization_5923/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5923/AssignMovingAvgAssignSubVariableOp@batch_normalization_5923_assignmovingavg_readvariableop_resource0batch_normalization_5923/AssignMovingAvg/mul:z:08^batch_normalization_5923/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5923/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5923_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5923/AssignMovingAvg_1/subSubAbatch_normalization_5923/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5923/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5923/AssignMovingAvg_1/mulMul2batch_normalization_5923/AssignMovingAvg_1/sub:z:09batch_normalization_5923/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5923/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5923_assignmovingavg_1_readvariableop_resource2batch_normalization_5923/AssignMovingAvg_1/mul:z:0:^batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5923/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5923/batchnorm/addAddV23batch_normalization_5923/moments/Squeeze_1:output:05batch_normalization_5923/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/RsqrtRsqrt*batch_normalization_5923/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5923/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5923_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5923/batchnorm/mul/mulMul,batch_normalization_5923/batchnorm/Rsqrt:y:0=batch_normalization_5923/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/mul_1Muldense_6561/Tanh:y:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5923/batchnorm/mul_2Mul1batch_normalization_5923/moments/Squeeze:output:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5923/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5923_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5923/batchnorm/subSub9batch_normalization_5923/batchnorm/ReadVariableOp:value:0,batch_normalization_5923/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/add_1AddV2,batch_normalization_5923/batchnorm/mul_1:z:0*batch_normalization_5923/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6562/MatMul/ReadVariableOpReadVariableOp)dense_6562_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6562/MatMulMatMul,batch_normalization_5923/batchnorm/add_1:z:0(dense_6562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6562/BiasAdd/ReadVariableOpReadVariableOp*dense_6562_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6562/BiasAddBiasAdddense_6562/MatMul:product:0)dense_6562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6562/ReluReludense_6562/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7batch_normalization_5924/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5924/moments/meanMeandense_6562/Relu:activations:0@batch_normalization_5924/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5924/moments/StopGradientStopGradient.batch_normalization_5924/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5924/moments/SquaredDifferenceSquaredDifferencedense_6562/Relu:activations:06batch_normalization_5924/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5924/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5924/moments/varianceMean6batch_normalization_5924/moments/SquaredDifference:z:0Dbatch_normalization_5924/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5924/moments/SqueezeSqueeze.batch_normalization_5924/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5924/moments/Squeeze_1Squeeze2batch_normalization_5924/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5924/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5924/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5924_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5924/AssignMovingAvg/subSub?batch_normalization_5924/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5924/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5924/AssignMovingAvg/mulMul0batch_normalization_5924/AssignMovingAvg/sub:z:07batch_normalization_5924/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5924/AssignMovingAvgAssignSubVariableOp@batch_normalization_5924_assignmovingavg_readvariableop_resource0batch_normalization_5924/AssignMovingAvg/mul:z:08^batch_normalization_5924/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5924/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5924_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5924/AssignMovingAvg_1/subSubAbatch_normalization_5924/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5924/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5924/AssignMovingAvg_1/mulMul2batch_normalization_5924/AssignMovingAvg_1/sub:z:09batch_normalization_5924/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5924/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5924_assignmovingavg_1_readvariableop_resource2batch_normalization_5924/AssignMovingAvg_1/mul:z:0:^batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5924/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5924/batchnorm/addAddV23batch_normalization_5924/moments/Squeeze_1:output:05batch_normalization_5924/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/RsqrtRsqrt*batch_normalization_5924/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5924/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5924_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5924/batchnorm/mul/mulMul,batch_normalization_5924/batchnorm/Rsqrt:y:0=batch_normalization_5924/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/mul_1Muldense_6562/Relu:activations:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5924/batchnorm/mul_2Mul1batch_normalization_5924/moments/Squeeze:output:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5924/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5924_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5924/batchnorm/subSub9batch_normalization_5924/batchnorm/ReadVariableOp:value:0,batch_normalization_5924/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/add_1AddV2,batch_normalization_5924/batchnorm/mul_1:z:0*batch_normalization_5924/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,batch_normalization_5924/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^batch_normalization_5921/AssignMovingAvg8^batch_normalization_5921/AssignMovingAvg/ReadVariableOp+^batch_normalization_5921/AssignMovingAvg_1:^batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5921/batchnorm/ReadVariableOp6^batch_normalization_5921/batchnorm/mul/ReadVariableOp)^batch_normalization_5922/AssignMovingAvg8^batch_normalization_5922/AssignMovingAvg/ReadVariableOp+^batch_normalization_5922/AssignMovingAvg_1:^batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5922/batchnorm/ReadVariableOp6^batch_normalization_5922/batchnorm/mul/ReadVariableOp)^batch_normalization_5923/AssignMovingAvg8^batch_normalization_5923/AssignMovingAvg/ReadVariableOp+^batch_normalization_5923/AssignMovingAvg_1:^batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5923/batchnorm/ReadVariableOp6^batch_normalization_5923/batchnorm/mul/ReadVariableOp)^batch_normalization_5924/AssignMovingAvg8^batch_normalization_5924/AssignMovingAvg/ReadVariableOp+^batch_normalization_5924/AssignMovingAvg_1:^batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5924/batchnorm/ReadVariableOp6^batch_normalization_5924/batchnorm/mul/ReadVariableOp"^dense_6559/BiasAdd/ReadVariableOp!^dense_6559/MatMul/ReadVariableOp"^dense_6560/BiasAdd/ReadVariableOp!^dense_6560/MatMul/ReadVariableOp"^dense_6561/BiasAdd/ReadVariableOp!^dense_6561/MatMul/ReadVariableOp"^dense_6562/BiasAdd/ReadVariableOp!^dense_6562/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_5921/AssignMovingAvg/ReadVariableOp7batch_normalization_5921/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5921/AssignMovingAvg_1*batch_normalization_5921/AssignMovingAvg_12T
(batch_normalization_5921/AssignMovingAvg(batch_normalization_5921/AssignMovingAvg2f
1batch_normalization_5921/batchnorm/ReadVariableOp1batch_normalization_5921/batchnorm/ReadVariableOp2n
5batch_normalization_5921/batchnorm/mul/ReadVariableOp5batch_normalization_5921/batchnorm/mul/ReadVariableOp2r
7batch_normalization_5922/AssignMovingAvg/ReadVariableOp7batch_normalization_5922/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5922/AssignMovingAvg_1*batch_normalization_5922/AssignMovingAvg_12T
(batch_normalization_5922/AssignMovingAvg(batch_normalization_5922/AssignMovingAvg2f
1batch_normalization_5922/batchnorm/ReadVariableOp1batch_normalization_5922/batchnorm/ReadVariableOp2n
5batch_normalization_5922/batchnorm/mul/ReadVariableOp5batch_normalization_5922/batchnorm/mul/ReadVariableOp2r
7batch_normalization_5923/AssignMovingAvg/ReadVariableOp7batch_normalization_5923/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5923/AssignMovingAvg_1*batch_normalization_5923/AssignMovingAvg_12T
(batch_normalization_5923/AssignMovingAvg(batch_normalization_5923/AssignMovingAvg2f
1batch_normalization_5923/batchnorm/ReadVariableOp1batch_normalization_5923/batchnorm/ReadVariableOp2n
5batch_normalization_5923/batchnorm/mul/ReadVariableOp5batch_normalization_5923/batchnorm/mul/ReadVariableOp2r
7batch_normalization_5924/AssignMovingAvg/ReadVariableOp7batch_normalization_5924/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5924/AssignMovingAvg_1*batch_normalization_5924/AssignMovingAvg_12T
(batch_normalization_5924/AssignMovingAvg(batch_normalization_5924/AssignMovingAvg2f
1batch_normalization_5924/batchnorm/ReadVariableOp1batch_normalization_5924/batchnorm/ReadVariableOp2n
5batch_normalization_5924/batchnorm/mul/ReadVariableOp5batch_normalization_5924/batchnorm/mul/ReadVariableOp2F
!dense_6559/BiasAdd/ReadVariableOp!dense_6559/BiasAdd/ReadVariableOp2D
 dense_6559/MatMul/ReadVariableOp dense_6559/MatMul/ReadVariableOp2F
!dense_6560/BiasAdd/ReadVariableOp!dense_6560/BiasAdd/ReadVariableOp2D
 dense_6560/MatMul/ReadVariableOp dense_6560/MatMul/ReadVariableOp2F
!dense_6561/BiasAdd/ReadVariableOp!dense_6561/BiasAdd/ReadVariableOp2D
 dense_6561/MatMul/ReadVariableOp dense_6561/MatMul/ReadVariableOp2F
!dense_6562/BiasAdd/ReadVariableOp!dense_6562/BiasAdd/ReadVariableOp2D
 dense_6562/MatMul/ReadVariableOp dense_6562/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
�&
�
<__inference_batch_normalization_5923_layer_call_fn_331781795

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781979

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
<__inference_batch_normalization_5922_layer_call_fn_331781685

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_dense_6561_layer_call_and_return_conditional_losses_331781761

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781739

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
<__inference_batch_normalization_5921_layer_call_fn_331781535

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_dense_6560_layer_call_fn_331781620

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference__wrapped_model_331780383

input_1051G
4model_1050_dense_6559_matmul_readvariableop_resource:	�C
5model_1050_dense_6559_biasadd_readvariableop_resource:S
Emodel_1050_batch_normalization_5921_batchnorm_readvariableop_resource:W
Imodel_1050_batch_normalization_5921_batchnorm_mul_readvariableop_resource:U
Gmodel_1050_batch_normalization_5921_batchnorm_readvariableop_1_resource:U
Gmodel_1050_batch_normalization_5921_batchnorm_readvariableop_2_resource:F
4model_1050_dense_6560_matmul_readvariableop_resource:C
5model_1050_dense_6560_biasadd_readvariableop_resource:S
Emodel_1050_batch_normalization_5922_batchnorm_readvariableop_resource:W
Imodel_1050_batch_normalization_5922_batchnorm_mul_readvariableop_resource:U
Gmodel_1050_batch_normalization_5922_batchnorm_readvariableop_1_resource:U
Gmodel_1050_batch_normalization_5922_batchnorm_readvariableop_2_resource:F
4model_1050_dense_6561_matmul_readvariableop_resource:C
5model_1050_dense_6561_biasadd_readvariableop_resource:S
Emodel_1050_batch_normalization_5923_batchnorm_readvariableop_resource:W
Imodel_1050_batch_normalization_5923_batchnorm_mul_readvariableop_resource:U
Gmodel_1050_batch_normalization_5923_batchnorm_readvariableop_1_resource:U
Gmodel_1050_batch_normalization_5923_batchnorm_readvariableop_2_resource:F
4model_1050_dense_6562_matmul_readvariableop_resource:C
5model_1050_dense_6562_biasadd_readvariableop_resource:S
Emodel_1050_batch_normalization_5924_batchnorm_readvariableop_resource:W
Imodel_1050_batch_normalization_5924_batchnorm_mul_readvariableop_resource:U
Gmodel_1050_batch_normalization_5924_batchnorm_readvariableop_1_resource:U
Gmodel_1050_batch_normalization_5924_batchnorm_readvariableop_2_resource:
identity��<model_1050/batch_normalization_5921/batchnorm/ReadVariableOp�>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_1�>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_2�@model_1050/batch_normalization_5921/batchnorm/mul/ReadVariableOp�<model_1050/batch_normalization_5922/batchnorm/ReadVariableOp�>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_1�>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_2�@model_1050/batch_normalization_5922/batchnorm/mul/ReadVariableOp�<model_1050/batch_normalization_5923/batchnorm/ReadVariableOp�>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_1�>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_2�@model_1050/batch_normalization_5923/batchnorm/mul/ReadVariableOp�<model_1050/batch_normalization_5924/batchnorm/ReadVariableOp�>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_1�>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_2�@model_1050/batch_normalization_5924/batchnorm/mul/ReadVariableOp�,model_1050/dense_6559/BiasAdd/ReadVariableOp�+model_1050/dense_6559/MatMul/ReadVariableOp�,model_1050/dense_6560/BiasAdd/ReadVariableOp�+model_1050/dense_6560/MatMul/ReadVariableOp�,model_1050/dense_6561/BiasAdd/ReadVariableOp�+model_1050/dense_6561/MatMul/ReadVariableOp�,model_1050/dense_6562/BiasAdd/ReadVariableOp�+model_1050/dense_6562/MatMul/ReadVariableOpi
model_1050/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
model_1050/flatten/ReshapeReshape
input_1051!model_1050/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
+model_1050/dense_6559/MatMul/ReadVariableOpReadVariableOp4model_1050_dense_6559_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_1050/dense_6559/MatMulMatMul#model_1050/flatten/Reshape:output:03model_1050/dense_6559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1050/dense_6559/BiasAdd/ReadVariableOpReadVariableOp5model_1050_dense_6559_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1050/dense_6559/BiasAddBiasAdd&model_1050/dense_6559/MatMul:product:04model_1050/dense_6559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<model_1050/batch_normalization_5921/batchnorm/ReadVariableOpReadVariableOpEmodel_1050_batch_normalization_5921_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0|
7model_1050/batch_normalization_5921/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_1050/batch_normalization_5921/batchnorm/addAddV2Dmodel_1050/batch_normalization_5921/batchnorm/ReadVariableOp:value:0@model_1050/batch_normalization_5921/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5921/batchnorm/RsqrtRsqrt5model_1050/batch_normalization_5921/batchnorm/add:z:0*
T0*
_output_shapes
:�
@model_1050/batch_normalization_5921/batchnorm/mul/ReadVariableOpReadVariableOpImodel_1050_batch_normalization_5921_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
5model_1050/batch_normalization_5921/batchnorm/mul/mulMul7model_1050/batch_normalization_5921/batchnorm/Rsqrt:y:0Hmodel_1050/batch_normalization_5921/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5921/batchnorm/mul_1Mul&model_1050/dense_6559/BiasAdd:output:09model_1050/batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_1050_batch_normalization_5921_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3model_1050/batch_normalization_5921/batchnorm/mul_2MulFmodel_1050/batch_normalization_5921/batchnorm/ReadVariableOp_1:value:09model_1050/batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_1050_batch_normalization_5921_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
1model_1050/batch_normalization_5921/batchnorm/subSubFmodel_1050/batch_normalization_5921/batchnorm/ReadVariableOp_2:value:07model_1050/batch_normalization_5921/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5921/batchnorm/add_1AddV27model_1050/batch_normalization_5921/batchnorm/mul_1:z:05model_1050/batch_normalization_5921/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
+model_1050/dense_6560/MatMul/ReadVariableOpReadVariableOp4model_1050_dense_6560_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_1050/dense_6560/MatMulMatMul7model_1050/batch_normalization_5921/batchnorm/add_1:z:03model_1050/dense_6560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1050/dense_6560/BiasAdd/ReadVariableOpReadVariableOp5model_1050_dense_6560_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1050/dense_6560/BiasAddBiasAdd&model_1050/dense_6560/MatMul:product:04model_1050/dense_6560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
model_1050/dense_6560/TanhTanh&model_1050/dense_6560/BiasAdd:output:0*
T0*'
_output_shapes
:����������
<model_1050/batch_normalization_5922/batchnorm/ReadVariableOpReadVariableOpEmodel_1050_batch_normalization_5922_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0|
7model_1050/batch_normalization_5922/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_1050/batch_normalization_5922/batchnorm/addAddV2Dmodel_1050/batch_normalization_5922/batchnorm/ReadVariableOp:value:0@model_1050/batch_normalization_5922/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5922/batchnorm/RsqrtRsqrt5model_1050/batch_normalization_5922/batchnorm/add:z:0*
T0*
_output_shapes
:�
@model_1050/batch_normalization_5922/batchnorm/mul/ReadVariableOpReadVariableOpImodel_1050_batch_normalization_5922_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
5model_1050/batch_normalization_5922/batchnorm/mul/mulMul7model_1050/batch_normalization_5922/batchnorm/Rsqrt:y:0Hmodel_1050/batch_normalization_5922/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5922/batchnorm/mul_1Mulmodel_1050/dense_6560/Tanh:y:09model_1050/batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_1050_batch_normalization_5922_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3model_1050/batch_normalization_5922/batchnorm/mul_2MulFmodel_1050/batch_normalization_5922/batchnorm/ReadVariableOp_1:value:09model_1050/batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_1050_batch_normalization_5922_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
1model_1050/batch_normalization_5922/batchnorm/subSubFmodel_1050/batch_normalization_5922/batchnorm/ReadVariableOp_2:value:07model_1050/batch_normalization_5922/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5922/batchnorm/add_1AddV27model_1050/batch_normalization_5922/batchnorm/mul_1:z:05model_1050/batch_normalization_5922/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
+model_1050/dense_6561/MatMul/ReadVariableOpReadVariableOp4model_1050_dense_6561_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_1050/dense_6561/MatMulMatMul7model_1050/batch_normalization_5922/batchnorm/add_1:z:03model_1050/dense_6561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1050/dense_6561/BiasAdd/ReadVariableOpReadVariableOp5model_1050_dense_6561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1050/dense_6561/BiasAddBiasAdd&model_1050/dense_6561/MatMul:product:04model_1050/dense_6561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
model_1050/dense_6561/TanhTanh&model_1050/dense_6561/BiasAdd:output:0*
T0*'
_output_shapes
:����������
<model_1050/batch_normalization_5923/batchnorm/ReadVariableOpReadVariableOpEmodel_1050_batch_normalization_5923_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0|
7model_1050/batch_normalization_5923/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_1050/batch_normalization_5923/batchnorm/addAddV2Dmodel_1050/batch_normalization_5923/batchnorm/ReadVariableOp:value:0@model_1050/batch_normalization_5923/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5923/batchnorm/RsqrtRsqrt5model_1050/batch_normalization_5923/batchnorm/add:z:0*
T0*
_output_shapes
:�
@model_1050/batch_normalization_5923/batchnorm/mul/ReadVariableOpReadVariableOpImodel_1050_batch_normalization_5923_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
5model_1050/batch_normalization_5923/batchnorm/mul/mulMul7model_1050/batch_normalization_5923/batchnorm/Rsqrt:y:0Hmodel_1050/batch_normalization_5923/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5923/batchnorm/mul_1Mulmodel_1050/dense_6561/Tanh:y:09model_1050/batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_1050_batch_normalization_5923_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3model_1050/batch_normalization_5923/batchnorm/mul_2MulFmodel_1050/batch_normalization_5923/batchnorm/ReadVariableOp_1:value:09model_1050/batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_1050_batch_normalization_5923_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
1model_1050/batch_normalization_5923/batchnorm/subSubFmodel_1050/batch_normalization_5923/batchnorm/ReadVariableOp_2:value:07model_1050/batch_normalization_5923/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5923/batchnorm/add_1AddV27model_1050/batch_normalization_5923/batchnorm/mul_1:z:05model_1050/batch_normalization_5923/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
+model_1050/dense_6562/MatMul/ReadVariableOpReadVariableOp4model_1050_dense_6562_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_1050/dense_6562/MatMulMatMul7model_1050/batch_normalization_5923/batchnorm/add_1:z:03model_1050/dense_6562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1050/dense_6562/BiasAdd/ReadVariableOpReadVariableOp5model_1050_dense_6562_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1050/dense_6562/BiasAddBiasAdd&model_1050/dense_6562/MatMul:product:04model_1050/dense_6562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
model_1050/dense_6562/ReluRelu&model_1050/dense_6562/BiasAdd:output:0*
T0*'
_output_shapes
:����������
<model_1050/batch_normalization_5924/batchnorm/ReadVariableOpReadVariableOpEmodel_1050_batch_normalization_5924_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0|
7model_1050/batch_normalization_5924/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_1050/batch_normalization_5924/batchnorm/addAddV2Dmodel_1050/batch_normalization_5924/batchnorm/ReadVariableOp:value:0@model_1050/batch_normalization_5924/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5924/batchnorm/RsqrtRsqrt5model_1050/batch_normalization_5924/batchnorm/add:z:0*
T0*
_output_shapes
:�
@model_1050/batch_normalization_5924/batchnorm/mul/ReadVariableOpReadVariableOpImodel_1050_batch_normalization_5924_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
5model_1050/batch_normalization_5924/batchnorm/mul/mulMul7model_1050/batch_normalization_5924/batchnorm/Rsqrt:y:0Hmodel_1050/batch_normalization_5924/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5924/batchnorm/mul_1Mul(model_1050/dense_6562/Relu:activations:09model_1050/batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_1050_batch_normalization_5924_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3model_1050/batch_normalization_5924/batchnorm/mul_2MulFmodel_1050/batch_normalization_5924/batchnorm/ReadVariableOp_1:value:09model_1050/batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_1050_batch_normalization_5924_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
1model_1050/batch_normalization_5924/batchnorm/subSubFmodel_1050/batch_normalization_5924/batchnorm/ReadVariableOp_2:value:07model_1050/batch_normalization_5924/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
3model_1050/batch_normalization_5924/batchnorm/add_1AddV27model_1050/batch_normalization_5924/batchnorm/mul_1:z:05model_1050/batch_normalization_5924/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
IdentityIdentity7model_1050/batch_normalization_5924/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp=^model_1050/batch_normalization_5921/batchnorm/ReadVariableOp?^model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_1?^model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_2A^model_1050/batch_normalization_5921/batchnorm/mul/ReadVariableOp=^model_1050/batch_normalization_5922/batchnorm/ReadVariableOp?^model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_1?^model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_2A^model_1050/batch_normalization_5922/batchnorm/mul/ReadVariableOp=^model_1050/batch_normalization_5923/batchnorm/ReadVariableOp?^model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_1?^model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_2A^model_1050/batch_normalization_5923/batchnorm/mul/ReadVariableOp=^model_1050/batch_normalization_5924/batchnorm/ReadVariableOp?^model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_1?^model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_2A^model_1050/batch_normalization_5924/batchnorm/mul/ReadVariableOp-^model_1050/dense_6559/BiasAdd/ReadVariableOp,^model_1050/dense_6559/MatMul/ReadVariableOp-^model_1050/dense_6560/BiasAdd/ReadVariableOp,^model_1050/dense_6560/MatMul/ReadVariableOp-^model_1050/dense_6561/BiasAdd/ReadVariableOp,^model_1050/dense_6561/MatMul/ReadVariableOp-^model_1050/dense_6562/BiasAdd/ReadVariableOp,^model_1050/dense_6562/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2�
>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_1>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_12�
>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_2>model_1050/batch_normalization_5921/batchnorm/ReadVariableOp_22|
<model_1050/batch_normalization_5921/batchnorm/ReadVariableOp<model_1050/batch_normalization_5921/batchnorm/ReadVariableOp2�
@model_1050/batch_normalization_5921/batchnorm/mul/ReadVariableOp@model_1050/batch_normalization_5921/batchnorm/mul/ReadVariableOp2�
>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_1>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_12�
>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_2>model_1050/batch_normalization_5922/batchnorm/ReadVariableOp_22|
<model_1050/batch_normalization_5922/batchnorm/ReadVariableOp<model_1050/batch_normalization_5922/batchnorm/ReadVariableOp2�
@model_1050/batch_normalization_5922/batchnorm/mul/ReadVariableOp@model_1050/batch_normalization_5922/batchnorm/mul/ReadVariableOp2�
>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_1>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_12�
>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_2>model_1050/batch_normalization_5923/batchnorm/ReadVariableOp_22|
<model_1050/batch_normalization_5923/batchnorm/ReadVariableOp<model_1050/batch_normalization_5923/batchnorm/ReadVariableOp2�
@model_1050/batch_normalization_5923/batchnorm/mul/ReadVariableOp@model_1050/batch_normalization_5923/batchnorm/mul/ReadVariableOp2�
>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_1>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_12�
>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_2>model_1050/batch_normalization_5924/batchnorm/ReadVariableOp_22|
<model_1050/batch_normalization_5924/batchnorm/ReadVariableOp<model_1050/batch_normalization_5924/batchnorm/ReadVariableOp2�
@model_1050/batch_normalization_5924/batchnorm/mul/ReadVariableOp@model_1050/batch_normalization_5924/batchnorm/mul/ReadVariableOp2\
,model_1050/dense_6559/BiasAdd/ReadVariableOp,model_1050/dense_6559/BiasAdd/ReadVariableOp2Z
+model_1050/dense_6559/MatMul/ReadVariableOp+model_1050/dense_6559/MatMul/ReadVariableOp2\
,model_1050/dense_6560/BiasAdd/ReadVariableOp,model_1050/dense_6560/BiasAdd/ReadVariableOp2Z
+model_1050/dense_6560/MatMul/ReadVariableOp+model_1050/dense_6560/MatMul/ReadVariableOp2\
,model_1050/dense_6561/BiasAdd/ReadVariableOp,model_1050/dense_6561/BiasAdd/ReadVariableOp2Z
+model_1050/dense_6561/MatMul/ReadVariableOp+model_1050/dense_6561/MatMul/ReadVariableOp2\
,model_1050/dense_6562/BiasAdd/ReadVariableOp,model_1050/dense_6562/BiasAdd/ReadVariableOp2Z
+model_1050/dense_6562/MatMul/ReadVariableOp+model_1050/dense_6562/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
�
�
'__inference_signature_wrapper_331781469

input_1051
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_1051unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_331780383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:)%
#
_user_specified_name	331781465:)%
#
_user_specified_name	331781463:)%
#
_user_specified_name	331781461:)%
#
_user_specified_name	331781459:)%
#
_user_specified_name	331781457:)%
#
_user_specified_name	331781455:)%
#
_user_specified_name	331781453:)%
#
_user_specified_name	331781451:)%
#
_user_specified_name	331781449:)%
#
_user_specified_name	331781447:)%
#
_user_specified_name	331781445:)%
#
_user_specified_name	331781443:)%
#
_user_specified_name	331781441:)%
#
_user_specified_name	331781439:)
%
#
_user_specified_name	331781437:)	%
#
_user_specified_name	331781435:)%
#
_user_specified_name	331781433:)%
#
_user_specified_name	331781431:)%
#
_user_specified_name	331781429:)%
#
_user_specified_name	331781427:)%
#
_user_specified_name	331781425:)%
#
_user_specified_name	331781423:)%
#
_user_specified_name	331781421:)%
#
_user_specified_name	331781419:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
�	
�
.__inference_dense_6559_layer_call_fn_331781491

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781999

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781609

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_dense_6559_layer_call_and_return_conditional_losses_331781501

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781719

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_dense_6562_layer_call_fn_331781880

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781849

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
<__inference_batch_normalization_5921_layer_call_fn_331781555

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
I__inference_model_1050_layer_call_and_return_conditional_losses_331781065

input_1051<
)dense_6559_matmul_readvariableop_resource:	�8
*dense_6559_biasadd_readvariableop_resource:H
:batch_normalization_5921_batchnorm_readvariableop_resource:L
>batch_normalization_5921_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5921_batchnorm_readvariableop_1_resource:J
<batch_normalization_5921_batchnorm_readvariableop_2_resource:;
)dense_6560_matmul_readvariableop_resource:8
*dense_6560_biasadd_readvariableop_resource:H
:batch_normalization_5922_batchnorm_readvariableop_resource:L
>batch_normalization_5922_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5922_batchnorm_readvariableop_1_resource:J
<batch_normalization_5922_batchnorm_readvariableop_2_resource:;
)dense_6561_matmul_readvariableop_resource:8
*dense_6561_biasadd_readvariableop_resource:H
:batch_normalization_5923_batchnorm_readvariableop_resource:L
>batch_normalization_5923_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5923_batchnorm_readvariableop_1_resource:J
<batch_normalization_5923_batchnorm_readvariableop_2_resource:;
)dense_6562_matmul_readvariableop_resource:8
*dense_6562_biasadd_readvariableop_resource:H
:batch_normalization_5924_batchnorm_readvariableop_resource:L
>batch_normalization_5924_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5924_batchnorm_readvariableop_1_resource:J
<batch_normalization_5924_batchnorm_readvariableop_2_resource:
identity��1batch_normalization_5921/batchnorm/ReadVariableOp�3batch_normalization_5921/batchnorm/ReadVariableOp_1�3batch_normalization_5921/batchnorm/ReadVariableOp_2�5batch_normalization_5921/batchnorm/mul/ReadVariableOp�1batch_normalization_5922/batchnorm/ReadVariableOp�3batch_normalization_5922/batchnorm/ReadVariableOp_1�3batch_normalization_5922/batchnorm/ReadVariableOp_2�5batch_normalization_5922/batchnorm/mul/ReadVariableOp�1batch_normalization_5923/batchnorm/ReadVariableOp�3batch_normalization_5923/batchnorm/ReadVariableOp_1�3batch_normalization_5923/batchnorm/ReadVariableOp_2�5batch_normalization_5923/batchnorm/mul/ReadVariableOp�1batch_normalization_5924/batchnorm/ReadVariableOp�3batch_normalization_5924/batchnorm/ReadVariableOp_1�3batch_normalization_5924/batchnorm/ReadVariableOp_2�5batch_normalization_5924/batchnorm/mul/ReadVariableOp�!dense_6559/BiasAdd/ReadVariableOp� dense_6559/MatMul/ReadVariableOp�!dense_6560/BiasAdd/ReadVariableOp� dense_6560/MatMul/ReadVariableOp�!dense_6561/BiasAdd/ReadVariableOp� dense_6561/MatMul/ReadVariableOp�!dense_6562/BiasAdd/ReadVariableOp� dense_6562/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  q
flatten/ReshapeReshape
input_1051flatten/Const:output:0*
T0*(
_output_shapes
:�����������
 dense_6559/MatMul/ReadVariableOpReadVariableOp)dense_6559_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_6559/MatMulMatMulflatten/Reshape:output:0(dense_6559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6559/BiasAdd/ReadVariableOpReadVariableOp*dense_6559_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6559/BiasAddBiasAdddense_6559/MatMul:product:0)dense_6559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1batch_normalization_5921/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5921_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5921/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5921/batchnorm/addAddV29batch_normalization_5921/batchnorm/ReadVariableOp:value:05batch_normalization_5921/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/RsqrtRsqrt*batch_normalization_5921/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5921/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5921_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5921/batchnorm/mul/mulMul,batch_normalization_5921/batchnorm/Rsqrt:y:0=batch_normalization_5921/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/mul_1Muldense_6559/BiasAdd:output:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5921/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5921_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5921/batchnorm/mul_2Mul;batch_normalization_5921/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5921/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5921_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5921/batchnorm/subSub;batch_normalization_5921/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5921/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/add_1AddV2,batch_normalization_5921/batchnorm/mul_1:z:0*batch_normalization_5921/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6560/MatMul/ReadVariableOpReadVariableOp)dense_6560_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6560/MatMulMatMul,batch_normalization_5921/batchnorm/add_1:z:0(dense_6560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6560/BiasAdd/ReadVariableOpReadVariableOp*dense_6560_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6560/BiasAddBiasAdddense_6560/MatMul:product:0)dense_6560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6560/TanhTanhdense_6560/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1batch_normalization_5922/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5922_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5922/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5922/batchnorm/addAddV29batch_normalization_5922/batchnorm/ReadVariableOp:value:05batch_normalization_5922/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/RsqrtRsqrt*batch_normalization_5922/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5922/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5922_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5922/batchnorm/mul/mulMul,batch_normalization_5922/batchnorm/Rsqrt:y:0=batch_normalization_5922/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/mul_1Muldense_6560/Tanh:y:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5922/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5922_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5922/batchnorm/mul_2Mul;batch_normalization_5922/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5922/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5922_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5922/batchnorm/subSub;batch_normalization_5922/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5922/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/add_1AddV2,batch_normalization_5922/batchnorm/mul_1:z:0*batch_normalization_5922/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6561/MatMul/ReadVariableOpReadVariableOp)dense_6561_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6561/MatMulMatMul,batch_normalization_5922/batchnorm/add_1:z:0(dense_6561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6561/BiasAdd/ReadVariableOpReadVariableOp*dense_6561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6561/BiasAddBiasAdddense_6561/MatMul:product:0)dense_6561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6561/TanhTanhdense_6561/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1batch_normalization_5923/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5923_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5923/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5923/batchnorm/addAddV29batch_normalization_5923/batchnorm/ReadVariableOp:value:05batch_normalization_5923/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/RsqrtRsqrt*batch_normalization_5923/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5923/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5923_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5923/batchnorm/mul/mulMul,batch_normalization_5923/batchnorm/Rsqrt:y:0=batch_normalization_5923/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/mul_1Muldense_6561/Tanh:y:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5923/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5923_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5923/batchnorm/mul_2Mul;batch_normalization_5923/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5923/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5923_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5923/batchnorm/subSub;batch_normalization_5923/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5923/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/add_1AddV2,batch_normalization_5923/batchnorm/mul_1:z:0*batch_normalization_5923/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6562/MatMul/ReadVariableOpReadVariableOp)dense_6562_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6562/MatMulMatMul,batch_normalization_5923/batchnorm/add_1:z:0(dense_6562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6562/BiasAdd/ReadVariableOpReadVariableOp*dense_6562_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6562/BiasAddBiasAdddense_6562/MatMul:product:0)dense_6562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6562/ReluReludense_6562/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1batch_normalization_5924/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5924_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5924/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5924/batchnorm/addAddV29batch_normalization_5924/batchnorm/ReadVariableOp:value:05batch_normalization_5924/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/RsqrtRsqrt*batch_normalization_5924/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5924/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5924_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5924/batchnorm/mul/mulMul,batch_normalization_5924/batchnorm/Rsqrt:y:0=batch_normalization_5924/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/mul_1Muldense_6562/Relu:activations:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5924/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5924_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5924/batchnorm/mul_2Mul;batch_normalization_5924/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5924/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5924_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5924/batchnorm/subSub;batch_normalization_5924/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5924/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/add_1AddV2,batch_normalization_5924/batchnorm/mul_1:z:0*batch_normalization_5924/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,batch_normalization_5924/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp2^batch_normalization_5921/batchnorm/ReadVariableOp4^batch_normalization_5921/batchnorm/ReadVariableOp_14^batch_normalization_5921/batchnorm/ReadVariableOp_26^batch_normalization_5921/batchnorm/mul/ReadVariableOp2^batch_normalization_5922/batchnorm/ReadVariableOp4^batch_normalization_5922/batchnorm/ReadVariableOp_14^batch_normalization_5922/batchnorm/ReadVariableOp_26^batch_normalization_5922/batchnorm/mul/ReadVariableOp2^batch_normalization_5923/batchnorm/ReadVariableOp4^batch_normalization_5923/batchnorm/ReadVariableOp_14^batch_normalization_5923/batchnorm/ReadVariableOp_26^batch_normalization_5923/batchnorm/mul/ReadVariableOp2^batch_normalization_5924/batchnorm/ReadVariableOp4^batch_normalization_5924/batchnorm/ReadVariableOp_14^batch_normalization_5924/batchnorm/ReadVariableOp_26^batch_normalization_5924/batchnorm/mul/ReadVariableOp"^dense_6559/BiasAdd/ReadVariableOp!^dense_6559/MatMul/ReadVariableOp"^dense_6560/BiasAdd/ReadVariableOp!^dense_6560/MatMul/ReadVariableOp"^dense_6561/BiasAdd/ReadVariableOp!^dense_6561/MatMul/ReadVariableOp"^dense_6562/BiasAdd/ReadVariableOp!^dense_6562/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization_5921/batchnorm/ReadVariableOp_13batch_normalization_5921/batchnorm/ReadVariableOp_12j
3batch_normalization_5921/batchnorm/ReadVariableOp_23batch_normalization_5921/batchnorm/ReadVariableOp_22f
1batch_normalization_5921/batchnorm/ReadVariableOp1batch_normalization_5921/batchnorm/ReadVariableOp2n
5batch_normalization_5921/batchnorm/mul/ReadVariableOp5batch_normalization_5921/batchnorm/mul/ReadVariableOp2j
3batch_normalization_5922/batchnorm/ReadVariableOp_13batch_normalization_5922/batchnorm/ReadVariableOp_12j
3batch_normalization_5922/batchnorm/ReadVariableOp_23batch_normalization_5922/batchnorm/ReadVariableOp_22f
1batch_normalization_5922/batchnorm/ReadVariableOp1batch_normalization_5922/batchnorm/ReadVariableOp2n
5batch_normalization_5922/batchnorm/mul/ReadVariableOp5batch_normalization_5922/batchnorm/mul/ReadVariableOp2j
3batch_normalization_5923/batchnorm/ReadVariableOp_13batch_normalization_5923/batchnorm/ReadVariableOp_12j
3batch_normalization_5923/batchnorm/ReadVariableOp_23batch_normalization_5923/batchnorm/ReadVariableOp_22f
1batch_normalization_5923/batchnorm/ReadVariableOp1batch_normalization_5923/batchnorm/ReadVariableOp2n
5batch_normalization_5923/batchnorm/mul/ReadVariableOp5batch_normalization_5923/batchnorm/mul/ReadVariableOp2j
3batch_normalization_5924/batchnorm/ReadVariableOp_13batch_normalization_5924/batchnorm/ReadVariableOp_12j
3batch_normalization_5924/batchnorm/ReadVariableOp_23batch_normalization_5924/batchnorm/ReadVariableOp_22f
1batch_normalization_5924/batchnorm/ReadVariableOp1batch_normalization_5924/batchnorm/ReadVariableOp2n
5batch_normalization_5924/batchnorm/mul/ReadVariableOp5batch_normalization_5924/batchnorm/mul/ReadVariableOp2F
!dense_6559/BiasAdd/ReadVariableOp!dense_6559/BiasAdd/ReadVariableOp2D
 dense_6559/MatMul/ReadVariableOp dense_6559/MatMul/ReadVariableOp2F
!dense_6560/BiasAdd/ReadVariableOp!dense_6560/BiasAdd/ReadVariableOp2D
 dense_6560/MatMul/ReadVariableOp dense_6560/MatMul/ReadVariableOp2F
!dense_6561/BiasAdd/ReadVariableOp!dense_6561/BiasAdd/ReadVariableOp2D
 dense_6561/MatMul/ReadVariableOp dense_6561/MatMul/ReadVariableOp2F
!dense_6562/BiasAdd/ReadVariableOp!dense_6562/BiasAdd/ReadVariableOp2D
 dense_6562/MatMul/ReadVariableOp dense_6562/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
�
G
+__inference_flatten_layer_call_fn_331781475

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781869

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
<__inference_batch_normalization_5922_layer_call_fn_331781665

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
I__inference_dense_6560_layer_call_and_return_conditional_losses_331781631

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
.__inference_model_1050_layer_call_fn_331781315

input_1051<
)dense_6559_matmul_readvariableop_resource:	�8
*dense_6559_biasadd_readvariableop_resource:H
:batch_normalization_5921_batchnorm_readvariableop_resource:L
>batch_normalization_5921_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5921_batchnorm_readvariableop_1_resource:J
<batch_normalization_5921_batchnorm_readvariableop_2_resource:;
)dense_6560_matmul_readvariableop_resource:8
*dense_6560_biasadd_readvariableop_resource:H
:batch_normalization_5922_batchnorm_readvariableop_resource:L
>batch_normalization_5922_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5922_batchnorm_readvariableop_1_resource:J
<batch_normalization_5922_batchnorm_readvariableop_2_resource:;
)dense_6561_matmul_readvariableop_resource:8
*dense_6561_biasadd_readvariableop_resource:H
:batch_normalization_5923_batchnorm_readvariableop_resource:L
>batch_normalization_5923_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5923_batchnorm_readvariableop_1_resource:J
<batch_normalization_5923_batchnorm_readvariableop_2_resource:;
)dense_6562_matmul_readvariableop_resource:8
*dense_6562_biasadd_readvariableop_resource:H
:batch_normalization_5924_batchnorm_readvariableop_resource:L
>batch_normalization_5924_batchnorm_mul_readvariableop_resource:J
<batch_normalization_5924_batchnorm_readvariableop_1_resource:J
<batch_normalization_5924_batchnorm_readvariableop_2_resource:
identity��1batch_normalization_5921/batchnorm/ReadVariableOp�3batch_normalization_5921/batchnorm/ReadVariableOp_1�3batch_normalization_5921/batchnorm/ReadVariableOp_2�5batch_normalization_5921/batchnorm/mul/ReadVariableOp�1batch_normalization_5922/batchnorm/ReadVariableOp�3batch_normalization_5922/batchnorm/ReadVariableOp_1�3batch_normalization_5922/batchnorm/ReadVariableOp_2�5batch_normalization_5922/batchnorm/mul/ReadVariableOp�1batch_normalization_5923/batchnorm/ReadVariableOp�3batch_normalization_5923/batchnorm/ReadVariableOp_1�3batch_normalization_5923/batchnorm/ReadVariableOp_2�5batch_normalization_5923/batchnorm/mul/ReadVariableOp�1batch_normalization_5924/batchnorm/ReadVariableOp�3batch_normalization_5924/batchnorm/ReadVariableOp_1�3batch_normalization_5924/batchnorm/ReadVariableOp_2�5batch_normalization_5924/batchnorm/mul/ReadVariableOp�!dense_6559/BiasAdd/ReadVariableOp� dense_6559/MatMul/ReadVariableOp�!dense_6560/BiasAdd/ReadVariableOp� dense_6560/MatMul/ReadVariableOp�!dense_6561/BiasAdd/ReadVariableOp� dense_6561/MatMul/ReadVariableOp�!dense_6562/BiasAdd/ReadVariableOp� dense_6562/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  q
flatten/ReshapeReshape
input_1051flatten/Const:output:0*
T0*(
_output_shapes
:�����������
 dense_6559/MatMul/ReadVariableOpReadVariableOp)dense_6559_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_6559/MatMulMatMulflatten/Reshape:output:0(dense_6559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6559/BiasAdd/ReadVariableOpReadVariableOp*dense_6559_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6559/BiasAddBiasAdddense_6559/MatMul:product:0)dense_6559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1batch_normalization_5921/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5921_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5921/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5921/batchnorm/addAddV29batch_normalization_5921/batchnorm/ReadVariableOp:value:05batch_normalization_5921/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/RsqrtRsqrt*batch_normalization_5921/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5921/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5921_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5921/batchnorm/mul/mulMul,batch_normalization_5921/batchnorm/Rsqrt:y:0=batch_normalization_5921/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/mul_1Muldense_6559/BiasAdd:output:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5921/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5921_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5921/batchnorm/mul_2Mul;batch_normalization_5921/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5921/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5921_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5921/batchnorm/subSub;batch_normalization_5921/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5921/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/add_1AddV2,batch_normalization_5921/batchnorm/mul_1:z:0*batch_normalization_5921/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6560/MatMul/ReadVariableOpReadVariableOp)dense_6560_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6560/MatMulMatMul,batch_normalization_5921/batchnorm/add_1:z:0(dense_6560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6560/BiasAdd/ReadVariableOpReadVariableOp*dense_6560_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6560/BiasAddBiasAdddense_6560/MatMul:product:0)dense_6560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6560/TanhTanhdense_6560/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1batch_normalization_5922/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5922_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5922/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5922/batchnorm/addAddV29batch_normalization_5922/batchnorm/ReadVariableOp:value:05batch_normalization_5922/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/RsqrtRsqrt*batch_normalization_5922/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5922/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5922_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5922/batchnorm/mul/mulMul,batch_normalization_5922/batchnorm/Rsqrt:y:0=batch_normalization_5922/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/mul_1Muldense_6560/Tanh:y:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5922/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5922_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5922/batchnorm/mul_2Mul;batch_normalization_5922/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5922/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5922_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5922/batchnorm/subSub;batch_normalization_5922/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5922/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/add_1AddV2,batch_normalization_5922/batchnorm/mul_1:z:0*batch_normalization_5922/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6561/MatMul/ReadVariableOpReadVariableOp)dense_6561_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6561/MatMulMatMul,batch_normalization_5922/batchnorm/add_1:z:0(dense_6561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6561/BiasAdd/ReadVariableOpReadVariableOp*dense_6561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6561/BiasAddBiasAdddense_6561/MatMul:product:0)dense_6561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6561/TanhTanhdense_6561/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1batch_normalization_5923/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5923_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5923/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5923/batchnorm/addAddV29batch_normalization_5923/batchnorm/ReadVariableOp:value:05batch_normalization_5923/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/RsqrtRsqrt*batch_normalization_5923/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5923/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5923_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5923/batchnorm/mul/mulMul,batch_normalization_5923/batchnorm/Rsqrt:y:0=batch_normalization_5923/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/mul_1Muldense_6561/Tanh:y:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5923/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5923_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5923/batchnorm/mul_2Mul;batch_normalization_5923/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5923/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5923_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5923/batchnorm/subSub;batch_normalization_5923/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5923/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/add_1AddV2,batch_normalization_5923/batchnorm/mul_1:z:0*batch_normalization_5923/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6562/MatMul/ReadVariableOpReadVariableOp)dense_6562_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6562/MatMulMatMul,batch_normalization_5923/batchnorm/add_1:z:0(dense_6562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6562/BiasAdd/ReadVariableOpReadVariableOp*dense_6562_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6562/BiasAddBiasAdddense_6562/MatMul:product:0)dense_6562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6562/ReluReludense_6562/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1batch_normalization_5924/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5924_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0q
,batch_normalization_5924/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5924/batchnorm/addAddV29batch_normalization_5924/batchnorm/ReadVariableOp:value:05batch_normalization_5924/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/RsqrtRsqrt*batch_normalization_5924/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5924/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5924_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5924/batchnorm/mul/mulMul,batch_normalization_5924/batchnorm/Rsqrt:y:0=batch_normalization_5924/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/mul_1Muldense_6562/Relu:activations:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
3batch_normalization_5924/batchnorm/ReadVariableOp_1ReadVariableOp<batch_normalization_5924_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(batch_normalization_5924/batchnorm/mul_2Mul;batch_normalization_5924/batchnorm/ReadVariableOp_1:value:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
3batch_normalization_5924/batchnorm/ReadVariableOp_2ReadVariableOp<batch_normalization_5924_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5924/batchnorm/subSub;batch_normalization_5924/batchnorm/ReadVariableOp_2:value:0,batch_normalization_5924/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/add_1AddV2,batch_normalization_5924/batchnorm/mul_1:z:0*batch_normalization_5924/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,batch_normalization_5924/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp2^batch_normalization_5921/batchnorm/ReadVariableOp4^batch_normalization_5921/batchnorm/ReadVariableOp_14^batch_normalization_5921/batchnorm/ReadVariableOp_26^batch_normalization_5921/batchnorm/mul/ReadVariableOp2^batch_normalization_5922/batchnorm/ReadVariableOp4^batch_normalization_5922/batchnorm/ReadVariableOp_14^batch_normalization_5922/batchnorm/ReadVariableOp_26^batch_normalization_5922/batchnorm/mul/ReadVariableOp2^batch_normalization_5923/batchnorm/ReadVariableOp4^batch_normalization_5923/batchnorm/ReadVariableOp_14^batch_normalization_5923/batchnorm/ReadVariableOp_26^batch_normalization_5923/batchnorm/mul/ReadVariableOp2^batch_normalization_5924/batchnorm/ReadVariableOp4^batch_normalization_5924/batchnorm/ReadVariableOp_14^batch_normalization_5924/batchnorm/ReadVariableOp_26^batch_normalization_5924/batchnorm/mul/ReadVariableOp"^dense_6559/BiasAdd/ReadVariableOp!^dense_6559/MatMul/ReadVariableOp"^dense_6560/BiasAdd/ReadVariableOp!^dense_6560/MatMul/ReadVariableOp"^dense_6561/BiasAdd/ReadVariableOp!^dense_6561/MatMul/ReadVariableOp"^dense_6562/BiasAdd/ReadVariableOp!^dense_6562/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization_5921/batchnorm/ReadVariableOp_13batch_normalization_5921/batchnorm/ReadVariableOp_12j
3batch_normalization_5921/batchnorm/ReadVariableOp_23batch_normalization_5921/batchnorm/ReadVariableOp_22f
1batch_normalization_5921/batchnorm/ReadVariableOp1batch_normalization_5921/batchnorm/ReadVariableOp2n
5batch_normalization_5921/batchnorm/mul/ReadVariableOp5batch_normalization_5921/batchnorm/mul/ReadVariableOp2j
3batch_normalization_5922/batchnorm/ReadVariableOp_13batch_normalization_5922/batchnorm/ReadVariableOp_12j
3batch_normalization_5922/batchnorm/ReadVariableOp_23batch_normalization_5922/batchnorm/ReadVariableOp_22f
1batch_normalization_5922/batchnorm/ReadVariableOp1batch_normalization_5922/batchnorm/ReadVariableOp2n
5batch_normalization_5922/batchnorm/mul/ReadVariableOp5batch_normalization_5922/batchnorm/mul/ReadVariableOp2j
3batch_normalization_5923/batchnorm/ReadVariableOp_13batch_normalization_5923/batchnorm/ReadVariableOp_12j
3batch_normalization_5923/batchnorm/ReadVariableOp_23batch_normalization_5923/batchnorm/ReadVariableOp_22f
1batch_normalization_5923/batchnorm/ReadVariableOp1batch_normalization_5923/batchnorm/ReadVariableOp2n
5batch_normalization_5923/batchnorm/mul/ReadVariableOp5batch_normalization_5923/batchnorm/mul/ReadVariableOp2j
3batch_normalization_5924/batchnorm/ReadVariableOp_13batch_normalization_5924/batchnorm/ReadVariableOp_12j
3batch_normalization_5924/batchnorm/ReadVariableOp_23batch_normalization_5924/batchnorm/ReadVariableOp_22f
1batch_normalization_5924/batchnorm/ReadVariableOp1batch_normalization_5924/batchnorm/ReadVariableOp2n
5batch_normalization_5924/batchnorm/mul/ReadVariableOp5batch_normalization_5924/batchnorm/mul/ReadVariableOp2F
!dense_6559/BiasAdd/ReadVariableOp!dense_6559/BiasAdd/ReadVariableOp2D
 dense_6559/MatMul/ReadVariableOp dense_6559/MatMul/ReadVariableOp2F
!dense_6560/BiasAdd/ReadVariableOp!dense_6560/BiasAdd/ReadVariableOp2D
 dense_6560/MatMul/ReadVariableOp dense_6560/MatMul/ReadVariableOp2F
!dense_6561/BiasAdd/ReadVariableOp!dense_6561/BiasAdd/ReadVariableOp2D
 dense_6561/MatMul/ReadVariableOp dense_6561/MatMul/ReadVariableOp2F
!dense_6562/BiasAdd/ReadVariableOp!dense_6562/BiasAdd/ReadVariableOp2D
 dense_6562/MatMul/ReadVariableOp dense_6562/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
��
�
.__inference_model_1050_layer_call_fn_331781218

input_1051<
)dense_6559_matmul_readvariableop_resource:	�8
*dense_6559_biasadd_readvariableop_resource:N
@batch_normalization_5921_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5921_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5921_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5921_batchnorm_readvariableop_resource:;
)dense_6560_matmul_readvariableop_resource:8
*dense_6560_biasadd_readvariableop_resource:N
@batch_normalization_5922_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5922_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5922_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5922_batchnorm_readvariableop_resource:;
)dense_6561_matmul_readvariableop_resource:8
*dense_6561_biasadd_readvariableop_resource:N
@batch_normalization_5923_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5923_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5923_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5923_batchnorm_readvariableop_resource:;
)dense_6562_matmul_readvariableop_resource:8
*dense_6562_biasadd_readvariableop_resource:N
@batch_normalization_5924_assignmovingavg_readvariableop_resource:P
Bbatch_normalization_5924_assignmovingavg_1_readvariableop_resource:L
>batch_normalization_5924_batchnorm_mul_readvariableop_resource:H
:batch_normalization_5924_batchnorm_readvariableop_resource:
identity��(batch_normalization_5921/AssignMovingAvg�7batch_normalization_5921/AssignMovingAvg/ReadVariableOp�*batch_normalization_5921/AssignMovingAvg_1�9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5921/batchnorm/ReadVariableOp�5batch_normalization_5921/batchnorm/mul/ReadVariableOp�(batch_normalization_5922/AssignMovingAvg�7batch_normalization_5922/AssignMovingAvg/ReadVariableOp�*batch_normalization_5922/AssignMovingAvg_1�9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5922/batchnorm/ReadVariableOp�5batch_normalization_5922/batchnorm/mul/ReadVariableOp�(batch_normalization_5923/AssignMovingAvg�7batch_normalization_5923/AssignMovingAvg/ReadVariableOp�*batch_normalization_5923/AssignMovingAvg_1�9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5923/batchnorm/ReadVariableOp�5batch_normalization_5923/batchnorm/mul/ReadVariableOp�(batch_normalization_5924/AssignMovingAvg�7batch_normalization_5924/AssignMovingAvg/ReadVariableOp�*batch_normalization_5924/AssignMovingAvg_1�9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp�1batch_normalization_5924/batchnorm/ReadVariableOp�5batch_normalization_5924/batchnorm/mul/ReadVariableOp�!dense_6559/BiasAdd/ReadVariableOp� dense_6559/MatMul/ReadVariableOp�!dense_6560/BiasAdd/ReadVariableOp� dense_6560/MatMul/ReadVariableOp�!dense_6561/BiasAdd/ReadVariableOp� dense_6561/MatMul/ReadVariableOp�!dense_6562/BiasAdd/ReadVariableOp� dense_6562/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  q
flatten/ReshapeReshape
input_1051flatten/Const:output:0*
T0*(
_output_shapes
:�����������
 dense_6559/MatMul/ReadVariableOpReadVariableOp)dense_6559_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_6559/MatMulMatMulflatten/Reshape:output:0(dense_6559/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6559/BiasAdd/ReadVariableOpReadVariableOp*dense_6559_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6559/BiasAddBiasAdddense_6559/MatMul:product:0)dense_6559/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7batch_normalization_5921/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5921/moments/meanMeandense_6559/BiasAdd:output:0@batch_normalization_5921/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5921/moments/StopGradientStopGradient.batch_normalization_5921/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5921/moments/SquaredDifferenceSquaredDifferencedense_6559/BiasAdd:output:06batch_normalization_5921/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5921/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5921/moments/varianceMean6batch_normalization_5921/moments/SquaredDifference:z:0Dbatch_normalization_5921/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5921/moments/SqueezeSqueeze.batch_normalization_5921/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5921/moments/Squeeze_1Squeeze2batch_normalization_5921/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5921/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5921/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5921_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5921/AssignMovingAvg/subSub?batch_normalization_5921/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5921/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5921/AssignMovingAvg/mulMul0batch_normalization_5921/AssignMovingAvg/sub:z:07batch_normalization_5921/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5921/AssignMovingAvgAssignSubVariableOp@batch_normalization_5921_assignmovingavg_readvariableop_resource0batch_normalization_5921/AssignMovingAvg/mul:z:08^batch_normalization_5921/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5921/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5921_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5921/AssignMovingAvg_1/subSubAbatch_normalization_5921/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5921/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5921/AssignMovingAvg_1/mulMul2batch_normalization_5921/AssignMovingAvg_1/sub:z:09batch_normalization_5921/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5921/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5921_assignmovingavg_1_readvariableop_resource2batch_normalization_5921/AssignMovingAvg_1/mul:z:0:^batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5921/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5921/batchnorm/addAddV23batch_normalization_5921/moments/Squeeze_1:output:05batch_normalization_5921/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/RsqrtRsqrt*batch_normalization_5921/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5921/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5921_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5921/batchnorm/mul/mulMul,batch_normalization_5921/batchnorm/Rsqrt:y:0=batch_normalization_5921/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/mul_1Muldense_6559/BiasAdd:output:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5921/batchnorm/mul_2Mul1batch_normalization_5921/moments/Squeeze:output:0.batch_normalization_5921/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5921/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5921_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5921/batchnorm/subSub9batch_normalization_5921/batchnorm/ReadVariableOp:value:0,batch_normalization_5921/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5921/batchnorm/add_1AddV2,batch_normalization_5921/batchnorm/mul_1:z:0*batch_normalization_5921/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6560/MatMul/ReadVariableOpReadVariableOp)dense_6560_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6560/MatMulMatMul,batch_normalization_5921/batchnorm/add_1:z:0(dense_6560/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6560/BiasAdd/ReadVariableOpReadVariableOp*dense_6560_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6560/BiasAddBiasAdddense_6560/MatMul:product:0)dense_6560/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6560/TanhTanhdense_6560/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7batch_normalization_5922/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5922/moments/meanMeandense_6560/Tanh:y:0@batch_normalization_5922/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5922/moments/StopGradientStopGradient.batch_normalization_5922/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5922/moments/SquaredDifferenceSquaredDifferencedense_6560/Tanh:y:06batch_normalization_5922/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5922/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5922/moments/varianceMean6batch_normalization_5922/moments/SquaredDifference:z:0Dbatch_normalization_5922/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5922/moments/SqueezeSqueeze.batch_normalization_5922/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5922/moments/Squeeze_1Squeeze2batch_normalization_5922/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5922/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5922/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5922_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5922/AssignMovingAvg/subSub?batch_normalization_5922/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5922/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5922/AssignMovingAvg/mulMul0batch_normalization_5922/AssignMovingAvg/sub:z:07batch_normalization_5922/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5922/AssignMovingAvgAssignSubVariableOp@batch_normalization_5922_assignmovingavg_readvariableop_resource0batch_normalization_5922/AssignMovingAvg/mul:z:08^batch_normalization_5922/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5922/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5922_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5922/AssignMovingAvg_1/subSubAbatch_normalization_5922/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5922/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5922/AssignMovingAvg_1/mulMul2batch_normalization_5922/AssignMovingAvg_1/sub:z:09batch_normalization_5922/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5922/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5922_assignmovingavg_1_readvariableop_resource2batch_normalization_5922/AssignMovingAvg_1/mul:z:0:^batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5922/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5922/batchnorm/addAddV23batch_normalization_5922/moments/Squeeze_1:output:05batch_normalization_5922/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/RsqrtRsqrt*batch_normalization_5922/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5922/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5922_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5922/batchnorm/mul/mulMul,batch_normalization_5922/batchnorm/Rsqrt:y:0=batch_normalization_5922/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/mul_1Muldense_6560/Tanh:y:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5922/batchnorm/mul_2Mul1batch_normalization_5922/moments/Squeeze:output:0.batch_normalization_5922/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5922/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5922_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5922/batchnorm/subSub9batch_normalization_5922/batchnorm/ReadVariableOp:value:0,batch_normalization_5922/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5922/batchnorm/add_1AddV2,batch_normalization_5922/batchnorm/mul_1:z:0*batch_normalization_5922/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6561/MatMul/ReadVariableOpReadVariableOp)dense_6561_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6561/MatMulMatMul,batch_normalization_5922/batchnorm/add_1:z:0(dense_6561/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6561/BiasAdd/ReadVariableOpReadVariableOp*dense_6561_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6561/BiasAddBiasAdddense_6561/MatMul:product:0)dense_6561/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6561/TanhTanhdense_6561/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7batch_normalization_5923/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5923/moments/meanMeandense_6561/Tanh:y:0@batch_normalization_5923/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5923/moments/StopGradientStopGradient.batch_normalization_5923/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5923/moments/SquaredDifferenceSquaredDifferencedense_6561/Tanh:y:06batch_normalization_5923/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5923/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5923/moments/varianceMean6batch_normalization_5923/moments/SquaredDifference:z:0Dbatch_normalization_5923/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5923/moments/SqueezeSqueeze.batch_normalization_5923/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5923/moments/Squeeze_1Squeeze2batch_normalization_5923/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5923/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5923/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5923_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5923/AssignMovingAvg/subSub?batch_normalization_5923/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5923/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5923/AssignMovingAvg/mulMul0batch_normalization_5923/AssignMovingAvg/sub:z:07batch_normalization_5923/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5923/AssignMovingAvgAssignSubVariableOp@batch_normalization_5923_assignmovingavg_readvariableop_resource0batch_normalization_5923/AssignMovingAvg/mul:z:08^batch_normalization_5923/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5923/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5923_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5923/AssignMovingAvg_1/subSubAbatch_normalization_5923/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5923/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5923/AssignMovingAvg_1/mulMul2batch_normalization_5923/AssignMovingAvg_1/sub:z:09batch_normalization_5923/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5923/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5923_assignmovingavg_1_readvariableop_resource2batch_normalization_5923/AssignMovingAvg_1/mul:z:0:^batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5923/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5923/batchnorm/addAddV23batch_normalization_5923/moments/Squeeze_1:output:05batch_normalization_5923/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/RsqrtRsqrt*batch_normalization_5923/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5923/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5923_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5923/batchnorm/mul/mulMul,batch_normalization_5923/batchnorm/Rsqrt:y:0=batch_normalization_5923/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/mul_1Muldense_6561/Tanh:y:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5923/batchnorm/mul_2Mul1batch_normalization_5923/moments/Squeeze:output:0.batch_normalization_5923/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5923/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5923_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5923/batchnorm/subSub9batch_normalization_5923/batchnorm/ReadVariableOp:value:0,batch_normalization_5923/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5923/batchnorm/add_1AddV2,batch_normalization_5923/batchnorm/mul_1:z:0*batch_normalization_5923/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
 dense_6562/MatMul/ReadVariableOpReadVariableOp)dense_6562_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_6562/MatMulMatMul,batch_normalization_5923/batchnorm/add_1:z:0(dense_6562/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_6562/BiasAdd/ReadVariableOpReadVariableOp*dense_6562_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_6562/BiasAddBiasAdddense_6562/MatMul:product:0)dense_6562/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_6562/ReluReludense_6562/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7batch_normalization_5924/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%batch_normalization_5924/moments/meanMeandense_6562/Relu:activations:0@batch_normalization_5924/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
-batch_normalization_5924/moments/StopGradientStopGradient.batch_normalization_5924/moments/mean:output:0*
T0*
_output_shapes

:�
2batch_normalization_5924/moments/SquaredDifferenceSquaredDifferencedense_6562/Relu:activations:06batch_normalization_5924/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
;batch_normalization_5924/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
)batch_normalization_5924/moments/varianceMean6batch_normalization_5924/moments/SquaredDifference:z:0Dbatch_normalization_5924/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization_5924/moments/SqueezeSqueeze.batch_normalization_5924/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
*batch_normalization_5924/moments/Squeeze_1Squeeze2batch_normalization_5924/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
.batch_normalization_5924/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_5924/AssignMovingAvg/ReadVariableOpReadVariableOp@batch_normalization_5924_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_5924/AssignMovingAvg/subSub?batch_normalization_5924/AssignMovingAvg/ReadVariableOp:value:01batch_normalization_5924/moments/Squeeze:output:0*
T0*
_output_shapes
:�
,batch_normalization_5924/AssignMovingAvg/mulMul0batch_normalization_5924/AssignMovingAvg/sub:z:07batch_normalization_5924/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_5924/AssignMovingAvgAssignSubVariableOp@batch_normalization_5924_assignmovingavg_readvariableop_resource0batch_normalization_5924/AssignMovingAvg/mul:z:08^batch_normalization_5924/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0u
0batch_normalization_5924/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOpReadVariableOpBbatch_normalization_5924_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
.batch_normalization_5924/AssignMovingAvg_1/subSubAbatch_normalization_5924/AssignMovingAvg_1/ReadVariableOp:value:03batch_normalization_5924/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
.batch_normalization_5924/AssignMovingAvg_1/mulMul2batch_normalization_5924/AssignMovingAvg_1/sub:z:09batch_normalization_5924/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
*batch_normalization_5924/AssignMovingAvg_1AssignSubVariableOpBbatch_normalization_5924_assignmovingavg_1_readvariableop_resource2batch_normalization_5924/AssignMovingAvg_1/mul:z:0:^batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0q
,batch_normalization_5924/batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&batch_normalization_5924/batchnorm/addAddV23batch_normalization_5924/moments/Squeeze_1:output:05batch_normalization_5924/batchnorm/add/Const:output:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/RsqrtRsqrt*batch_normalization_5924/batchnorm/add:z:0*
T0*
_output_shapes
:�
5batch_normalization_5924/batchnorm/mul/ReadVariableOpReadVariableOp>batch_normalization_5924_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_5924/batchnorm/mul/mulMul,batch_normalization_5924/batchnorm/Rsqrt:y:0=batch_normalization_5924/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/mul_1Muldense_6562/Relu:activations:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*'
_output_shapes
:����������
(batch_normalization_5924/batchnorm/mul_2Mul1batch_normalization_5924/moments/Squeeze:output:0.batch_normalization_5924/batchnorm/mul/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_5924/batchnorm/ReadVariableOpReadVariableOp:batch_normalization_5924_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_5924/batchnorm/subSub9batch_normalization_5924/batchnorm/ReadVariableOp:value:0,batch_normalization_5924/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(batch_normalization_5924/batchnorm/add_1AddV2,batch_normalization_5924/batchnorm/mul_1:z:0*batch_normalization_5924/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,batch_normalization_5924/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^batch_normalization_5921/AssignMovingAvg8^batch_normalization_5921/AssignMovingAvg/ReadVariableOp+^batch_normalization_5921/AssignMovingAvg_1:^batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5921/batchnorm/ReadVariableOp6^batch_normalization_5921/batchnorm/mul/ReadVariableOp)^batch_normalization_5922/AssignMovingAvg8^batch_normalization_5922/AssignMovingAvg/ReadVariableOp+^batch_normalization_5922/AssignMovingAvg_1:^batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5922/batchnorm/ReadVariableOp6^batch_normalization_5922/batchnorm/mul/ReadVariableOp)^batch_normalization_5923/AssignMovingAvg8^batch_normalization_5923/AssignMovingAvg/ReadVariableOp+^batch_normalization_5923/AssignMovingAvg_1:^batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5923/batchnorm/ReadVariableOp6^batch_normalization_5923/batchnorm/mul/ReadVariableOp)^batch_normalization_5924/AssignMovingAvg8^batch_normalization_5924/AssignMovingAvg/ReadVariableOp+^batch_normalization_5924/AssignMovingAvg_1:^batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp2^batch_normalization_5924/batchnorm/ReadVariableOp6^batch_normalization_5924/batchnorm/mul/ReadVariableOp"^dense_6559/BiasAdd/ReadVariableOp!^dense_6559/MatMul/ReadVariableOp"^dense_6560/BiasAdd/ReadVariableOp!^dense_6560/MatMul/ReadVariableOp"^dense_6561/BiasAdd/ReadVariableOp!^dense_6561/MatMul/ReadVariableOp"^dense_6562/BiasAdd/ReadVariableOp!^dense_6562/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_5921/AssignMovingAvg/ReadVariableOp7batch_normalization_5921/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5921/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5921/AssignMovingAvg_1*batch_normalization_5921/AssignMovingAvg_12T
(batch_normalization_5921/AssignMovingAvg(batch_normalization_5921/AssignMovingAvg2f
1batch_normalization_5921/batchnorm/ReadVariableOp1batch_normalization_5921/batchnorm/ReadVariableOp2n
5batch_normalization_5921/batchnorm/mul/ReadVariableOp5batch_normalization_5921/batchnorm/mul/ReadVariableOp2r
7batch_normalization_5922/AssignMovingAvg/ReadVariableOp7batch_normalization_5922/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5922/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5922/AssignMovingAvg_1*batch_normalization_5922/AssignMovingAvg_12T
(batch_normalization_5922/AssignMovingAvg(batch_normalization_5922/AssignMovingAvg2f
1batch_normalization_5922/batchnorm/ReadVariableOp1batch_normalization_5922/batchnorm/ReadVariableOp2n
5batch_normalization_5922/batchnorm/mul/ReadVariableOp5batch_normalization_5922/batchnorm/mul/ReadVariableOp2r
7batch_normalization_5923/AssignMovingAvg/ReadVariableOp7batch_normalization_5923/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5923/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5923/AssignMovingAvg_1*batch_normalization_5923/AssignMovingAvg_12T
(batch_normalization_5923/AssignMovingAvg(batch_normalization_5923/AssignMovingAvg2f
1batch_normalization_5923/batchnorm/ReadVariableOp1batch_normalization_5923/batchnorm/ReadVariableOp2n
5batch_normalization_5923/batchnorm/mul/ReadVariableOp5batch_normalization_5923/batchnorm/mul/ReadVariableOp2r
7batch_normalization_5924/AssignMovingAvg/ReadVariableOp7batch_normalization_5924/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp9batch_normalization_5924/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_5924/AssignMovingAvg_1*batch_normalization_5924/AssignMovingAvg_12T
(batch_normalization_5924/AssignMovingAvg(batch_normalization_5924/AssignMovingAvg2f
1batch_normalization_5924/batchnorm/ReadVariableOp1batch_normalization_5924/batchnorm/ReadVariableOp2n
5batch_normalization_5924/batchnorm/mul/ReadVariableOp5batch_normalization_5924/batchnorm/mul/ReadVariableOp2F
!dense_6559/BiasAdd/ReadVariableOp!dense_6559/BiasAdd/ReadVariableOp2D
 dense_6559/MatMul/ReadVariableOp dense_6559/MatMul/ReadVariableOp2F
!dense_6560/BiasAdd/ReadVariableOp!dense_6560/BiasAdd/ReadVariableOp2D
 dense_6560/MatMul/ReadVariableOp dense_6560/MatMul/ReadVariableOp2F
!dense_6561/BiasAdd/ReadVariableOp!dense_6561/BiasAdd/ReadVariableOp2D
 dense_6561/MatMul/ReadVariableOp dense_6561/MatMul/ReadVariableOp2F
!dense_6562/BiasAdd/ReadVariableOp!dense_6562/BiasAdd/ReadVariableOp2D
 dense_6562/MatMul/ReadVariableOp dense_6562/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
�
�
<__inference_batch_normalization_5923_layer_call_fn_331781815

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:{
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0v
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
<__inference_batch_normalization_5924_layer_call_fn_331781925

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0X
batchnorm/add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/Const:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0x
batchnorm/mul/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:g
batchnorm/mul_1Mulinputsbatchnorm/mul/mul:z:0*
T0*'
_output_shapes
:���������l
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�2
saver_filename:0Identity_124:0Identity_1878"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
B

input_10514
serving_default_input_1051:0����������L
batch_normalization_59240
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance"
_tf_keras_layer
�
 0
!1
)2
*3
+4
,5
36
47
<8
=9
>10
?11
F12
G13
O14
P15
Q16
R17
Y18
Z19
b20
c21
d22
e23"
trackable_list_wrapper
�
 0
!1
)2
*3
34
45
<6
=7
F8
G9
O10
P11
Y12
Z13
b14
c15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ktrace_0
ltrace_12�
.__inference_model_1050_layer_call_fn_331781218
.__inference_model_1050_layer_call_fn_331781315�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zktrace_0zltrace_1
�
mtrace_0
ntrace_12�
I__inference_model_1050_layer_call_and_return_conditional_losses_331780968
I__inference_model_1050_layer_call_and_return_conditional_losses_331781065�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zmtrace_0zntrace_1
�B�
$__inference__wrapped_model_331780383
input_1051"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
o
_variables
p_iterations
q_learning_rate
r_index_dict
s
_momentums
t_velocities
u_update_step_xla"
experimentalOptimizer
,
vserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
|trace_02�
+__inference_flatten_layer_call_fn_331781475�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z|trace_0
�
}trace_02�
F__inference_flatten_layer_call_and_return_conditional_losses_331781481�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_6559_layer_call_fn_331781491�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_6559_layer_call_and_return_conditional_losses_331781501�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"	�2dense_6559/kernel
:2dense_6559/bias
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
<__inference_batch_normalization_5921_layer_call_fn_331781535
<__inference_batch_normalization_5921_layer_call_fn_331781555�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781589
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781609�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
,:*2batch_normalization_5921/gamma
+:)2batch_normalization_5921/beta
4:2 (2$batch_normalization_5921/moving_mean
8:6 (2(batch_normalization_5921/moving_variance
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_6560_layer_call_fn_331781620�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_6560_layer_call_and_return_conditional_losses_331781631�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!2dense_6560/kernel
:2dense_6560/bias
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
<__inference_batch_normalization_5922_layer_call_fn_331781665
<__inference_batch_normalization_5922_layer_call_fn_331781685�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781719
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781739�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
,:*2batch_normalization_5922/gamma
+:)2batch_normalization_5922/beta
4:2 (2$batch_normalization_5922/moving_mean
8:6 (2(batch_normalization_5922/moving_variance
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_6561_layer_call_fn_331781750�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_6561_layer_call_and_return_conditional_losses_331781761�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!2dense_6561/kernel
:2dense_6561/bias
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
<__inference_batch_normalization_5923_layer_call_fn_331781795
<__inference_batch_normalization_5923_layer_call_fn_331781815�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781849
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781869�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
,:*2batch_normalization_5923/gamma
+:)2batch_normalization_5923/beta
4:2 (2$batch_normalization_5923/moving_mean
8:6 (2(batch_normalization_5923/moving_variance
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_dense_6562_layer_call_fn_331781880�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_dense_6562_layer_call_and_return_conditional_losses_331781891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!2dense_6562/kernel
:2dense_6562/bias
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
<__inference_batch_normalization_5924_layer_call_fn_331781925
<__inference_batch_normalization_5924_layer_call_fn_331781945�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781979
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781999�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
,:*2batch_normalization_5924/gamma
+:)2batch_normalization_5924/beta
4:2 (2$batch_normalization_5924/moving_mean
8:6 (2(batch_normalization_5924/moving_variance
X
+0
,1
>2
?3
Q4
R5
d6
e7"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_model_1050_layer_call_fn_331781218
input_1051"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_model_1050_layer_call_fn_331781315
input_1051"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_model_1050_layer_call_and_return_conditional_losses_331780968
input_1051"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_model_1050_layer_call_and_return_conditional_losses_331781065
input_1051"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
p0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
'__inference_signature_wrapper_331781469
input_1051"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_flatten_layer_call_fn_331781475inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_layer_call_and_return_conditional_losses_331781481inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_6559_layer_call_fn_331781491inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_6559_layer_call_and_return_conditional_losses_331781501inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_batch_normalization_5921_layer_call_fn_331781535inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_batch_normalization_5921_layer_call_fn_331781555inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781589inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781609inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_6560_layer_call_fn_331781620inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_6560_layer_call_and_return_conditional_losses_331781631inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_batch_normalization_5922_layer_call_fn_331781665inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_batch_normalization_5922_layer_call_fn_331781685inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781719inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781739inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_6561_layer_call_fn_331781750inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_6561_layer_call_and_return_conditional_losses_331781761inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_batch_normalization_5923_layer_call_fn_331781795inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_batch_normalization_5923_layer_call_fn_331781815inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781849inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781869inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_6562_layer_call_fn_331781880inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_dense_6562_layer_call_and_return_conditional_losses_331781891inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_batch_normalization_5924_layer_call_fn_331781925inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_batch_normalization_5924_layer_call_fn_331781945inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781979inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781999inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
):'	�2Adam/m/dense_6559/kernel
):'	�2Adam/v/dense_6559/kernel
": 2Adam/m/dense_6559/bias
": 2Adam/v/dense_6559/bias
1:/2%Adam/m/batch_normalization_5921/gamma
1:/2%Adam/v/batch_normalization_5921/gamma
0:.2$Adam/m/batch_normalization_5921/beta
0:.2$Adam/v/batch_normalization_5921/beta
(:&2Adam/m/dense_6560/kernel
(:&2Adam/v/dense_6560/kernel
": 2Adam/m/dense_6560/bias
": 2Adam/v/dense_6560/bias
1:/2%Adam/m/batch_normalization_5922/gamma
1:/2%Adam/v/batch_normalization_5922/gamma
0:.2$Adam/m/batch_normalization_5922/beta
0:.2$Adam/v/batch_normalization_5922/beta
(:&2Adam/m/dense_6561/kernel
(:&2Adam/v/dense_6561/kernel
": 2Adam/m/dense_6561/bias
": 2Adam/v/dense_6561/bias
1:/2%Adam/m/batch_normalization_5923/gamma
1:/2%Adam/v/batch_normalization_5923/gamma
0:.2$Adam/m/batch_normalization_5923/beta
0:.2$Adam/v/batch_normalization_5923/beta
(:&2Adam/m/dense_6562/kernel
(:&2Adam/v/dense_6562/kernel
": 2Adam/m/dense_6562/bias
": 2Adam/v/dense_6562/bias
1:/2%Adam/m/batch_normalization_5924/gamma
1:/2%Adam/v/batch_normalization_5924/gamma
0:.2$Adam/m/batch_normalization_5924/beta
0:.2$Adam/v/batch_normalization_5924/beta
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
$__inference__wrapped_model_331780383� !,)+*34?<>=FGROQPYZebdc4�1
*�'
%�"

input_1051����������
� "S�P
N
batch_normalization_59242�/
batch_normalization_5924����������
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781589m+,)*7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
W__inference_batch_normalization_5921_layer_call_and_return_conditional_losses_331781609m,)+*7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
<__inference_batch_normalization_5921_layer_call_fn_331781535b+,)*7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
<__inference_batch_normalization_5921_layer_call_fn_331781555b,)+*7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781719m>?<=7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
W__inference_batch_normalization_5922_layer_call_and_return_conditional_losses_331781739m?<>=7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
<__inference_batch_normalization_5922_layer_call_fn_331781665b>?<=7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
<__inference_batch_normalization_5922_layer_call_fn_331781685b?<>=7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781849mQROP7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
W__inference_batch_normalization_5923_layer_call_and_return_conditional_losses_331781869mROQP7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
<__inference_batch_normalization_5923_layer_call_fn_331781795bQROP7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
<__inference_batch_normalization_5923_layer_call_fn_331781815bROQP7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781979mdebc7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
W__inference_batch_normalization_5924_layer_call_and_return_conditional_losses_331781999mebdc7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
<__inference_batch_normalization_5924_layer_call_fn_331781925bdebc7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
<__inference_batch_normalization_5924_layer_call_fn_331781945bebdc7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
I__inference_dense_6559_layer_call_and_return_conditional_losses_331781501d !0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
.__inference_dense_6559_layer_call_fn_331781491Y !0�-
&�#
!�
inputs����������
� "!�
unknown����������
I__inference_dense_6560_layer_call_and_return_conditional_losses_331781631c34/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
.__inference_dense_6560_layer_call_fn_331781620X34/�,
%�"
 �
inputs���������
� "!�
unknown����������
I__inference_dense_6561_layer_call_and_return_conditional_losses_331781761cFG/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
.__inference_dense_6561_layer_call_fn_331781750XFG/�,
%�"
 �
inputs���������
� "!�
unknown����������
I__inference_dense_6562_layer_call_and_return_conditional_losses_331781891cYZ/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
.__inference_dense_6562_layer_call_fn_331781880XYZ/�,
%�"
 �
inputs���������
� "!�
unknown����������
F__inference_flatten_layer_call_and_return_conditional_losses_331781481a0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_flatten_layer_call_fn_331781475V0�-
&�#
!�
inputs����������
� ""�
unknown�����������
I__inference_model_1050_layer_call_and_return_conditional_losses_331780968� !+,)*34>?<=FGQROPYZdebc<�9
2�/
%�"

input_1051����������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_model_1050_layer_call_and_return_conditional_losses_331781065� !,)+*34?<>=FGROQPYZebdc<�9
2�/
%�"

input_1051����������
p 

 
� ",�)
"�
tensor_0���������
� �
.__inference_model_1050_layer_call_fn_331781218{ !+,)*34>?<=FGQROPYZdebc<�9
2�/
%�"

input_1051����������
p

 
� "!�
unknown����������
.__inference_model_1050_layer_call_fn_331781315{ !,)+*34?<>=FGROQPYZebdc<�9
2�/
%�"

input_1051����������
p 

 
� "!�
unknown����������
'__inference_signature_wrapper_331781469� !,)+*34?<>=FGROQPYZebdcB�?
� 
8�5
3

input_1051%�"

input_1051����������"S�P
N
batch_normalization_59242�/
batch_normalization_5924���������