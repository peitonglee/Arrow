
for i in [0, len(edges)]:
    if i == 0:
        current_edge = edges[0]   # 当前边
    else:
        last_final_edges_end_point = final_edges[-1][1]   # 获取上一个加入最终轮廓边的末尾点
        current_edge = find_next_anchor_edge(last_final_edges_end_point)   # 使用上一个轮廓边的末尾点寻找当前边
    endif

    for i in [0, len(current_nodes)]:   # 当前边产生的所有节点，这些节点构成的边必定共线
        for j in [0, len(current_nodes)]:  
            tmp_edges.push(current_nodes[i], current_nodes[j])   # 两两一对产生一个临时边
        endfor
    endfor
    
    selected_edge = max(IOU(tmp_edges, current_edge))   # 临时边中与当前边IOU最大的边为这条边的轮廓边
    final_edges.push(selected_edge)   # 将IOU最大的边加入轮廓集
endfor
