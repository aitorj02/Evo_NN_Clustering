��
��
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.12v2.13.0-17-gf841394b1b78��
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
Adam/v/dense_3683/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3683/bias
}
*Adam/v/dense_3683/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3683/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3683/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3683/bias
}
*Adam/m/dense_3683/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3683/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3683/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/dense_3683/kernel
�
,Adam/v/dense_3683/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3683/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_3683/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/dense_3683/kernel
�
,Adam/m/dense_3683/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3683/kernel*
_output_shapes
:	�*
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
v
dense_3683/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3683/bias
o
#dense_3683/bias/Read/ReadVariableOpReadVariableOpdense_3683/bias*
_output_shapes
:*
dtype0

dense_3683/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namedense_3683/kernel
x
%dense_3683/kernel/Read/ReadVariableOpReadVariableOpdense_3683/kernel*
_output_shapes
:	�*
dtype0

serving_default_input_1051Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1051dense_3683/kerneldense_3683/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_157480697

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
layer_with_weights-0
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*

 trace_0
!trace_1* 

"trace_0
#trace_1* 
* 
�
$
_variables
%_iterations
&_learning_rate
'_index_dict
(
_momentums
)_velocities
*_update_step_xla*

+serving_default* 
* 
* 
* 
�
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

1trace_0* 

2trace_0* 

0
1*

0
1*
* 
�
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

8trace_0* 

9trace_0* 
a[
VARIABLE_VALUEdense_3683/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3683/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

:0
;1*
* 
* 
* 
* 
* 
* 
'
%0
<1
=2
>3
?4*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

<0
>1*

=0
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
* 
8
@	variables
A	keras_api
	Btotal
	Ccount*
H
D	variables
E	keras_api
	Ftotal
	Gcount
H
_fn_kwargs*
c]
VARIABLE_VALUEAdam/m/dense_3683/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_3683/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3683/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3683/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

@	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

D	variables*
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
Read/DisableCopyOnReadDisableCopyOnReaddense_3683/kernel"/device:CPU:0
u
Read/ReadVariableOpReadVariableOpdense_3683/kernel"/device:CPU:0*
_output_shapes
:	�*
dtype0
b
IdentityIdentityRead/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
Y

Identity_1IdentityIdentity"/device:CPU:0*
T0*
_output_shapes
:	�
M
Read_1/DisableCopyOnReadDisableCopyOnReaddense_3683/bias"/device:CPU:0
p
Read_1/ReadVariableOpReadVariableOpdense_3683/bias"/device:CPU:0*
_output_shapes
:*
dtype0
a

Identity_2IdentityRead_1/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
V

Identity_3Identity
Identity_2"/device:CPU:0*
T0*
_output_shapes
:
G
Read_2/DisableCopyOnReadDisableCopyOnRead	iteration"/device:CPU:0
f
Read_2/ReadVariableOpReadVariableOp	iteration"/device:CPU:0*
_output_shapes
: *
dtype0	
]

Identity_4IdentityRead_2/ReadVariableOp"/device:CPU:0*
T0	*
_output_shapes
: 
R

Identity_5Identity
Identity_4"/device:CPU:0*
T0	*
_output_shapes
: 
K
Read_3/DisableCopyOnReadDisableCopyOnReadlearning_rate"/device:CPU:0
j
Read_3/ReadVariableOpReadVariableOplearning_rate"/device:CPU:0*
_output_shapes
: *
dtype0
]

Identity_6IdentityRead_3/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
R

Identity_7Identity
Identity_6"/device:CPU:0*
T0*
_output_shapes
: 
V
Read_4/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_3683/kernel"/device:CPU:0
~
Read_4/ReadVariableOpReadVariableOpAdam/m/dense_3683/kernel"/device:CPU:0*
_output_shapes
:	�*
dtype0
f

Identity_8IdentityRead_4/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
[

Identity_9Identity
Identity_8"/device:CPU:0*
T0*
_output_shapes
:	�
V
Read_5/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_3683/kernel"/device:CPU:0
~
Read_5/ReadVariableOpReadVariableOpAdam/v/dense_3683/kernel"/device:CPU:0*
_output_shapes
:	�*
dtype0
g
Identity_10IdentityRead_5/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	�
]
Identity_11IdentityIdentity_10"/device:CPU:0*
T0*
_output_shapes
:	�
T
Read_6/DisableCopyOnReadDisableCopyOnReadAdam/m/dense_3683/bias"/device:CPU:0
w
Read_6/ReadVariableOpReadVariableOpAdam/m/dense_3683/bias"/device:CPU:0*
_output_shapes
:*
dtype0
b
Identity_12IdentityRead_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_13IdentityIdentity_12"/device:CPU:0*
T0*
_output_shapes
:
T
Read_7/DisableCopyOnReadDisableCopyOnReadAdam/v/dense_3683/bias"/device:CPU:0
w
Read_7/ReadVariableOpReadVariableOpAdam/v/dense_3683/bias"/device:CPU:0*
_output_shapes
:*
dtype0
b
Identity_14IdentityRead_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
X
Identity_15IdentityIdentity_14"/device:CPU:0*
T0*
_output_shapes
:
E
Read_8/DisableCopyOnReadDisableCopyOnReadtotal_1"/device:CPU:0
d
Read_8/ReadVariableOpReadVariableOptotal_1"/device:CPU:0*
_output_shapes
: *
dtype0
^
Identity_16IdentityRead_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
T
Identity_17IdentityIdentity_16"/device:CPU:0*
T0*
_output_shapes
: 
E
Read_9/DisableCopyOnReadDisableCopyOnReadcount_1"/device:CPU:0
d
Read_9/ReadVariableOpReadVariableOpcount_1"/device:CPU:0*
_output_shapes
: *
dtype0
^
Identity_18IdentityRead_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
T
Identity_19IdentityIdentity_18"/device:CPU:0*
T0*
_output_shapes
: 
D
Read_10/DisableCopyOnReadDisableCopyOnReadtotal"/device:CPU:0
c
Read_10/ReadVariableOpReadVariableOptotal"/device:CPU:0*
_output_shapes
: *
dtype0
_
Identity_20IdentityRead_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
T
Identity_21IdentityIdentity_20"/device:CPU:0*
T0*
_output_shapes
: 
D
Read_11/DisableCopyOnReadDisableCopyOnReadcount"/device:CPU:0
c
Read_11/ReadVariableOpReadVariableOpcount"/device:CPU:0*
_output_shapes
: *
dtype0
_
Identity_22IdentityRead_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
: 
T
Identity_23IdentityIdentity_22"/device:CPU:0*
T0*
_output_shapes
: 
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 
�
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices
Identity_1
Identity_3
Identity_5
Identity_7
Identity_9Identity_11Identity_13Identity_15Identity_17Identity_19Identity_21Identity_23Const"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtypes
2	
�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
�
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0*&
 _has_manual_control_dependencies(
l
Identity_24Identitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 
�
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	
T
Identity_25Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOpAssignVariableOpdense_3683/kernelIdentity_25"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_26IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_1AssignVariableOpdense_3683/biasIdentity_26"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_27IdentityRestoreV2:2"/device:CPU:0*
T0	*
_output_shapes
:
�
AssignVariableOp_2AssignVariableOp	iterationIdentity_27"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0	
V
Identity_28IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_3AssignVariableOplearning_rateIdentity_28"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_29IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_4AssignVariableOpAdam/m/dense_3683/kernelIdentity_29"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_30IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_5AssignVariableOpAdam/v/dense_3683/kernelIdentity_30"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_31IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_6AssignVariableOpAdam/m/dense_3683/biasIdentity_31"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_32IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_7AssignVariableOpAdam/v/dense_3683/biasIdentity_32"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_33IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_8AssignVariableOptotal_1Identity_33"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
V
Identity_34IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_9AssignVariableOpcount_1Identity_34"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_35IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:

AssignVariableOp_10AssignVariableOptotalIdentity_35"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
W
Identity_36IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:

AssignVariableOp_11AssignVariableOpcountIdentity_36"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
E
NoOp_1NoOp"/device:CPU:0*&
 _has_manual_control_dependencies(
�
Identity_37Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: �o
�
�
.__inference_model_1050_layer_call_fn_157480638

input_1051<
)dense_3683_matmul_readvariableop_resource:	�8
*dense_3683_biasadd_readvariableop_resource:
identity��!dense_3683/BiasAdd/ReadVariableOp� dense_3683/MatMul/ReadVariableOp^
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
 dense_3683/MatMul/ReadVariableOpReadVariableOp)dense_3683_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3683/MatMulMatMulflatten/Reshape:output:0(dense_3683/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_3683/BiasAdd/ReadVariableOpReadVariableOp*dense_3683_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3683/BiasAddBiasAdddense_3683/MatMul:product:0)dense_3683/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
dense_3683/SigmoidSigmoiddense_3683/BiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentitydense_3683/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������i
NoOpNoOp"^dense_3683/BiasAdd/ReadVariableOp!^dense_3683/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2F
!dense_3683/BiasAdd/ReadVariableOp!dense_3683/BiasAdd/ReadVariableOp2D
 dense_3683/MatMul/ReadVariableOp dense_3683/MatMul/ReadVariableOp:($
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
�
�
'__inference_signature_wrapper_157480697

input_1051
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_1051unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_157480599o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:)%
#
_user_specified_name	157480693:)%
#
_user_specified_name	157480691:T P
(
_output_shapes
:����������
$
_user_specified_name
input_1051
�

�
I__inference_dense_3683_layer_call_and_return_conditional_losses_157480731

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
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
�
G
+__inference_flatten_layer_call_fn_157480703

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
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_157480709

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
�
�
.__inference_model_1050_layer_call_fn_157480651

input_1051<
)dense_3683_matmul_readvariableop_resource:	�8
*dense_3683_biasadd_readvariableop_resource:
identity��!dense_3683/BiasAdd/ReadVariableOp� dense_3683/MatMul/ReadVariableOp^
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
 dense_3683/MatMul/ReadVariableOpReadVariableOp)dense_3683_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3683/MatMulMatMulflatten/Reshape:output:0(dense_3683/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_3683/BiasAdd/ReadVariableOpReadVariableOp*dense_3683_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3683/BiasAddBiasAdddense_3683/MatMul:product:0)dense_3683/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
dense_3683/SigmoidSigmoiddense_3683/BiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentitydense_3683/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������i
NoOpNoOp"^dense_3683/BiasAdd/ReadVariableOp!^dense_3683/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2F
!dense_3683/BiasAdd/ReadVariableOp!dense_3683/BiasAdd/ReadVariableOp2D
 dense_3683/MatMul/ReadVariableOp dense_3683/MatMul/ReadVariableOp:($
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
�
�
I__inference_model_1050_layer_call_and_return_conditional_losses_157480612

input_1051<
)dense_3683_matmul_readvariableop_resource:	�8
*dense_3683_biasadd_readvariableop_resource:
identity��!dense_3683/BiasAdd/ReadVariableOp� dense_3683/MatMul/ReadVariableOp^
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
 dense_3683/MatMul/ReadVariableOpReadVariableOp)dense_3683_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3683/MatMulMatMulflatten/Reshape:output:0(dense_3683/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_3683/BiasAdd/ReadVariableOpReadVariableOp*dense_3683_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3683/BiasAddBiasAdddense_3683/MatMul:product:0)dense_3683/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
dense_3683/SigmoidSigmoiddense_3683/BiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentitydense_3683/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������i
NoOpNoOp"^dense_3683/BiasAdd/ReadVariableOp!^dense_3683/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2F
!dense_3683/BiasAdd/ReadVariableOp!dense_3683/BiasAdd/ReadVariableOp2D
 dense_3683/MatMul/ReadVariableOp dense_3683/MatMul/ReadVariableOp:($
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
�
�
$__inference__wrapped_model_157480599

input_1051G
4model_1050_dense_3683_matmul_readvariableop_resource:	�C
5model_1050_dense_3683_biasadd_readvariableop_resource:
identity��,model_1050/dense_3683/BiasAdd/ReadVariableOp�+model_1050/dense_3683/MatMul/ReadVariableOpi
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
+model_1050/dense_3683/MatMul/ReadVariableOpReadVariableOp4model_1050_dense_3683_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_1050/dense_3683/MatMulMatMul#model_1050/flatten/Reshape:output:03model_1050/dense_3683/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1050/dense_3683/BiasAdd/ReadVariableOpReadVariableOp5model_1050_dense_3683_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1050/dense_3683/BiasAddBiasAdd&model_1050/dense_3683/MatMul:product:04model_1050/dense_3683/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_1050/dense_3683/SigmoidSigmoid&model_1050/dense_3683/BiasAdd:output:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!model_1050/dense_3683/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������
NoOpNoOp-^model_1050/dense_3683/BiasAdd/ReadVariableOp,^model_1050/dense_3683/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2\
,model_1050/dense_3683/BiasAdd/ReadVariableOp,model_1050/dense_3683/BiasAdd/ReadVariableOp2Z
+model_1050/dense_3683/MatMul/ReadVariableOp+model_1050/dense_3683/MatMul/ReadVariableOp:($
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
�

�
.__inference_dense_3683_layer_call_fn_157480720

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
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
�
�
I__inference_model_1050_layer_call_and_return_conditional_losses_157480625

input_1051<
)dense_3683_matmul_readvariableop_resource:	�8
*dense_3683_biasadd_readvariableop_resource:
identity��!dense_3683/BiasAdd/ReadVariableOp� dense_3683/MatMul/ReadVariableOp^
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
 dense_3683/MatMul/ReadVariableOpReadVariableOp)dense_3683_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_3683/MatMulMatMulflatten/Reshape:output:0(dense_3683/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dense_3683/BiasAdd/ReadVariableOpReadVariableOp*dense_3683_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3683/BiasAddBiasAdddense_3683/MatMul:product:0)dense_3683/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
dense_3683/SigmoidSigmoiddense_3683/BiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentitydense_3683/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������i
NoOpNoOp"^dense_3683/BiasAdd/ReadVariableOp!^dense_3683/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2F
!dense_3683/BiasAdd/ReadVariableOp!dense_3683/BiasAdd/ReadVariableOp2D
 dense_3683/MatMul/ReadVariableOp dense_3683/MatMul/ReadVariableOp:($
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
input_1051"�0
saver_filename:0Identity_24:0Identity_378"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
B

input_10514
serving_default_input_1051:0����������>

dense_36830
StatefulPartitionedCall:0���������tensorflow/serving/predict:�K
�
layer-0
layer-1
layer_with_weights-0
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
 trace_0
!trace_12�
.__inference_model_1050_layer_call_fn_157480638
.__inference_model_1050_layer_call_fn_157480651�
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
 z trace_0z!trace_1
�
"trace_0
#trace_12�
I__inference_model_1050_layer_call_and_return_conditional_losses_157480612
I__inference_model_1050_layer_call_and_return_conditional_losses_157480625�
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
 z"trace_0z#trace_1
�B�
$__inference__wrapped_model_157480599
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
$
_variables
%_iterations
&_learning_rate
'_index_dict
(
_momentums
)_velocities
*_update_step_xla"
experimentalOptimizer
,
+serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
1trace_02�
+__inference_flatten_layer_call_fn_157480703�
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
 z1trace_0
�
2trace_02�
F__inference_flatten_layer_call_and_return_conditional_losses_157480709�
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
 z2trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
8trace_02�
.__inference_dense_3683_layer_call_fn_157480720�
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
 z8trace_0
�
9trace_02�
I__inference_dense_3683_layer_call_and_return_conditional_losses_157480731�
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
 z9trace_0
$:"	�2dense_3683/kernel
:2dense_3683/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_model_1050_layer_call_fn_157480638
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
.__inference_model_1050_layer_call_fn_157480651
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
I__inference_model_1050_layer_call_and_return_conditional_losses_157480612
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
I__inference_model_1050_layer_call_and_return_conditional_losses_157480625
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
C
%0
<1
=2
>3
?4"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
.
<0
>1"
trackable_list_wrapper
.
=0
?1"
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
'__inference_signature_wrapper_157480697
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
+__inference_flatten_layer_call_fn_157480703inputs"�
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
F__inference_flatten_layer_call_and_return_conditional_losses_157480709inputs"�
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
.__inference_dense_3683_layer_call_fn_157480720inputs"�
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
I__inference_dense_3683_layer_call_and_return_conditional_losses_157480731inputs"�
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
N
@	variables
A	keras_api
	Btotal
	Ccount"
_tf_keras_metric
^
D	variables
E	keras_api
	Ftotal
	Gcount
H
_fn_kwargs"
_tf_keras_metric
):'	�2Adam/m/dense_3683/kernel
):'	�2Adam/v/dense_3683/kernel
": 2Adam/m/dense_3683/bias
": 2Adam/v/dense_3683/bias
.
B0
C1"
trackable_list_wrapper
-
@	variables"
_generic_user_object
:  (2total
:  (2count
.
F0
G1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
$__inference__wrapped_model_157480599s4�1
*�'
%�"

input_1051����������
� "7�4
2

dense_3683$�!

dense_3683����������
I__inference_dense_3683_layer_call_and_return_conditional_losses_157480731d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
.__inference_dense_3683_layer_call_fn_157480720Y0�-
&�#
!�
inputs����������
� "!�
unknown����������
F__inference_flatten_layer_call_and_return_conditional_losses_157480709a0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_flatten_layer_call_fn_157480703V0�-
&�#
!�
inputs����������
� ""�
unknown�����������
I__inference_model_1050_layer_call_and_return_conditional_losses_157480612p<�9
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
I__inference_model_1050_layer_call_and_return_conditional_losses_157480625p<�9
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
.__inference_model_1050_layer_call_fn_157480638e<�9
2�/
%�"

input_1051����������
p

 
� "!�
unknown����������
.__inference_model_1050_layer_call_fn_157480651e<�9
2�/
%�"

input_1051����������
p 

 
� "!�
unknown����������
'__inference_signature_wrapper_157480697�B�?
� 
8�5
3

input_1051%�"

input_1051����������"7�4
2

dense_3683$�!

dense_3683���������