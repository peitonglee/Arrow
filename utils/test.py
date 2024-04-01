import math
def find_intersection(range1, range2):
    A, B = range1
    C, D = range2
    # 如果两个范围没有交集，则返回空范围
    if B < C or D < A:
        return None
    # 否则，返回交集范围
    return max(A, C), min(B, D)

def find_union(range1, range2):
    x1, x2 = range1
    x3, x4 = range2
    # 计算并集的起始和结束
    union_start = min(x1, x3)
    union_end = max(x2, x4)
    # 返回并集范围
    return union_start, union_end


line1 = [1, 1, 2, 0]
line2 = [0, 2, 3, -1]

xx1, yy1, xx2, yy2 = line1
xx3, yy3, xx4, yy4 = line2
    
if xx1 == xx2 and xx1 == xx3 and xx3 == xx4:
    x1, x2, x3, x4 = yy1, yy2, yy3, yy4
    y1, y2, y3, y4 = xx1, xx2, xx3, xx4
else:
    x1, x2, x3, x4 = xx1, xx2, xx3, xx4
    y1, y2, y3, y4 = yy1, yy2, yy3, yy4  
     
range1 = [x1, x2]
range2 = [x3, x4]

range1.sort()
range2.sort()
intersection = find_intersection(range1, range2)


kk = (y2-y1)/(x2-x1)

if kk == 0:
    length = abs(intersection[1]-intersection[0])
else:
    length = math.sqrt((1+kk**2)*abs(intersection[1]-intersection[0])**2)
print("交集长度为:", length)

union = find_union(range1, range2)
kk = (y2-y1)/(x2-x1)


if kk == 0:
    uni_length = abs(union[1]-union[0])
else:
    uni_length = math.sqrt((1+kk**2)*abs(union[1]-union[0])**2)
print("并集长度为:", uni_length)