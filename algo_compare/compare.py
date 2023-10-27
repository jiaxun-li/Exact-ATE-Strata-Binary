import Balanced
import LiDing
import Origin
import Partial_balanced

matrix_n=[
[10,10,10,10],[1,19,0,20]
]
rep=10000

print(Balanced.confidence_interval_permute(matrix_n,replication=rep))
print(Partial_balanced.confidence_interval_permute(matrix_n,replication=rep))
print(LiDing.confidence_interval_permute(matrix_n,replication=rep))
print(Origin.confidence_interval_permute(matrix_n,replication=rep))


# matrix_n=[
#     [2,6,4,4],
#     [2,6,0,8]
# ]
#[-10, 10, 254]
# [-9, 10, 26073]
# [-9, 10, 30541]
# [-9, 10, 62937]


# matrix_n=[
#     [2,6,4,4],
#     [2,6,0,8],
#     [2,4,4,2]
# ]
# [-15, 9, 4822]
# [-16, 10, 1665657]
# [-16, 8, 4780515]
# [-15, 8, 10132857]



# matrix_n=[
#     [2,3,4,1],
#     [0,4,0,4],
#     [2,4,4,2],
#     [5,1,3,3]
# ]
# [-15, 8, 12813]
# [-16, 10, 6301890]
# [-16, 10, 35393259]
# [-16, 9, 56800800]

# matrix_n=[
#     [2,3,1,4],
#     [2,7,5,4],
#     [0,10,1,9]
# ]
# [-18, 6, 14638]
# [-17, 6, 1361920]
# [-18, 8, 4541833]
# [-17, 7, 9292800]