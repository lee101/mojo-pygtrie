from std.sys.info import simd_width_of


comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


@export("mpg_fill_i64")
def mpg_fill_i64(
    destination_addr: Int,
    length: Int,
    value: Int64,
) abi("C"):
    var destination = IPtr(unsafe_from_address=destination_addr)
    comptime W = simd_width_of[DType.int64]()
    var values = SIMD[DType.int64, W](value)
    var index = 0
    while index + W <= length:
        destination.store(index, values)
        index += W
    while index < length:
        destination.store(index, value)
        index += 1


@export("mpg_grow_i64")
def mpg_grow_i64(
    source_addr: Int,
    destination_addr: Int,
    old_length: Int,
    new_length: Int,
    fill: Int64,
) abi("C"):
    var source = IPtr(unsafe_from_address=source_addr)
    var destination = IPtr(unsafe_from_address=destination_addr)
    comptime W = simd_width_of[DType.int64]()
    var index = 0
    while index + W <= old_length:
        destination.store(index, source.load[width=W](index))
        index += W
    while index < old_length:
        destination.store(index, source.load(index))
        index += 1
    var values = SIMD[DType.int64, W](fill)
    while index + W <= new_length:
        destination.store(index, values)
        index += W
    while index < new_length:
        destination.store(index, fill)
        index += 1


def hash_index(node: Int64, token: Int64, capacity: Int) -> Int:
    return Int((node * 1000003 + token * 9176) & Int64(capacity - 1))


def find_child(
    hash_nodes: IPtr,
    hash_tokens: IPtr,
    hash_children: IPtr,
    capacity: Int,
    node: Int64,
    token: Int64,
) -> Int64:
    var slot = hash_index(node, token, capacity)
    var probes = 0
    while probes < capacity:
        var owner = hash_nodes.load(slot)
        if owner == -1:
            return -1
        if owner == node and hash_tokens.load(slot) == token:
            return hash_children.load(slot)
        slot = (slot + 1) & (capacity - 1)
        probes += 1
    return -1


def add_hash_edge(
    hash_nodes: IPtr,
    hash_tokens: IPtr,
    hash_children: IPtr,
    capacity: Int,
    node: Int64,
    token: Int64,
    child: Int64,
):
    var slot = hash_index(node, token, capacity)
    var tombstone = -1
    while True:
        var owner = hash_nodes.load(slot)
        if owner == -1:
            if tombstone >= 0:
                slot = tombstone
            hash_nodes.store(slot, node)
            hash_tokens.store(slot, token)
            hash_children.store(slot, child)
            return
        if owner == -2 and tombstone < 0:
            tombstone = slot
        slot = (slot + 1) & (capacity - 1)


def find_path(
    hash_nodes: IPtr,
    hash_tokens: IPtr,
    hash_children: IPtr,
    hash_capacity: Int,
    tokens: IPtr,
    offsets: IPtr,
    idx: Int,
) -> Int64:
    var node = Int64(0)
    var pos = Int(offsets.load(idx))
    var stop = Int(offsets.load(idx + 1))
    while pos < stop and node >= 0:
        node = find_child(
            hash_nodes,
            hash_tokens,
            hash_children,
            hash_capacity,
            node,
            tokens.load(pos),
        )
        pos += 1
    return node


def longest_path(
    hash_nodes: IPtr,
    hash_tokens: IPtr,
    hash_children: IPtr,
    hash_capacity: Int,
    has_value: BPtr,
    tokens: IPtr,
    offsets: IPtr,
    idx: Int,
) -> Int64:
    var node = Int64(0)
    var best = Int64(-1)
    if has_value.load(0) != 0:
        best = 0
    var pos = Int(offsets.load(idx))
    var stop = Int(offsets.load(idx + 1))
    while pos < stop:
        node = find_child(
            hash_nodes,
            hash_tokens,
            hash_children,
            hash_capacity,
            node,
            tokens.load(pos),
        )
        if node < 0:
            break
        if has_value.load(Int(node)) != 0:
            best = node
        pos += 1
    return best


def insert_path(
    hash_nodes: IPtr,
    hash_tokens: IPtr,
    hash_children: IPtr,
    hash_capacity: Int,
    first_edge: IPtr,
    last_edge: IPtr,
    edge_tokens: IPtr,
    edge_children: IPtr,
    edge_next: IPtr,
    parents: IPtr,
    parent_tokens: IPtr,
    state: IPtr,
    tokens: IPtr,
    start: Int,
    stop: Int,
) -> Int64:
    var node = Int64(0)
    var pos = start
    var node_count = state.load(0)
    var edge_count = state.load(1)
    while pos < stop:
        var token = tokens.load(pos)
        var child = find_child(
            hash_nodes, hash_tokens, hash_children, hash_capacity, node, token
        )
        if child < 0:
            child = node_count
            node_count += 1
            var edge = edge_count
            edge_count += 1
            first_edge.store(Int(child), -1)
            last_edge.store(Int(child), -1)
            parents.store(Int(child), node)
            parent_tokens.store(Int(child), token)
            edge_tokens.store(Int(edge), token)
            edge_children.store(Int(edge), child)
            edge_next.store(Int(edge), -1)
            var tail = last_edge.load(Int(node))
            if tail < 0:
                first_edge.store(Int(node), edge)
            else:
                edge_next.store(Int(tail), edge)
            last_edge.store(Int(node), edge)
            add_hash_edge(
                hash_nodes,
                hash_tokens,
                hash_children,
                hash_capacity,
                node,
                token,
                child,
            )
        node = child
        pos += 1
    state.store(0, node_count)
    state.store(1, edge_count)
    return node


@export("mpg_find")
def mpg_find(
    hash_nodes_addr: Int,
    hash_tokens_addr: Int,
    hash_children_addr: Int,
    hash_capacity: Int,
    tokens_addr: Int,
    length: Int,
) abi("C") -> Int:
    var hash_nodes = IPtr(unsafe_from_address=hash_nodes_addr)
    var hash_tokens = IPtr(unsafe_from_address=hash_tokens_addr)
    var hash_children = IPtr(unsafe_from_address=hash_children_addr)
    var tokens = IPtr(unsafe_from_address=tokens_addr)
    var node = Int64(0)
    for pos in range(length):
        node = find_child(
            hash_nodes,
            hash_tokens,
            hash_children,
            hash_capacity,
            node,
            tokens.load(pos),
        )
        if node < 0:
            return -1
    return Int(node)


@export("mpg_insert")
def mpg_insert(
    hash_nodes_addr: Int,
    hash_tokens_addr: Int,
    hash_children_addr: Int,
    hash_capacity: Int,
    first_edge_addr: Int,
    last_edge_addr: Int,
    edge_tokens_addr: Int,
    edge_children_addr: Int,
    edge_next_addr: Int,
    parents_addr: Int,
    parent_tokens_addr: Int,
    state_addr: Int,
    tokens_addr: Int,
    length: Int,
) abi("C") -> Int:
    return Int(
        insert_path(
            IPtr(unsafe_from_address=hash_nodes_addr),
            IPtr(unsafe_from_address=hash_tokens_addr),
            IPtr(unsafe_from_address=hash_children_addr),
            hash_capacity,
            IPtr(unsafe_from_address=first_edge_addr),
            IPtr(unsafe_from_address=last_edge_addr),
            IPtr(unsafe_from_address=edge_tokens_addr),
            IPtr(unsafe_from_address=edge_children_addr),
            IPtr(unsafe_from_address=edge_next_addr),
            IPtr(unsafe_from_address=parents_addr),
            IPtr(unsafe_from_address=parent_tokens_addr),
            IPtr(unsafe_from_address=state_addr),
            IPtr(unsafe_from_address=tokens_addr),
            0,
            length,
        )
    )


@export("mpg_bulk_insert")
def mpg_bulk_insert(
    hash_nodes_addr: Int,
    hash_tokens_addr: Int,
    hash_children_addr: Int,
    hash_capacity: Int,
    first_edge_addr: Int,
    last_edge_addr: Int,
    edge_tokens_addr: Int,
    edge_children_addr: Int,
    edge_next_addr: Int,
    parents_addr: Int,
    parent_tokens_addr: Int,
    state_addr: Int,
    tokens_addr: Int,
    offsets_addr: Int,
    count: Int,
    result_addr: Int,
) abi("C"):
    var hash_nodes = IPtr(unsafe_from_address=hash_nodes_addr)
    var hash_tokens = IPtr(unsafe_from_address=hash_tokens_addr)
    var hash_children = IPtr(unsafe_from_address=hash_children_addr)
    var first_edge = IPtr(unsafe_from_address=first_edge_addr)
    var last_edge = IPtr(unsafe_from_address=last_edge_addr)
    var edge_tokens = IPtr(unsafe_from_address=edge_tokens_addr)
    var edge_children = IPtr(unsafe_from_address=edge_children_addr)
    var edge_next = IPtr(unsafe_from_address=edge_next_addr)
    var parents = IPtr(unsafe_from_address=parents_addr)
    var parent_tokens = IPtr(unsafe_from_address=parent_tokens_addr)
    var state = IPtr(unsafe_from_address=state_addr)
    var tokens = IPtr(unsafe_from_address=tokens_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    for idx in range(count):
        result.store(
            idx,
            insert_path(
                hash_nodes,
                hash_tokens,
                hash_children,
                hash_capacity,
                first_edge,
                last_edge,
                edge_tokens,
                edge_children,
                edge_next,
                parents,
                parent_tokens,
                state,
                tokens,
                Int(offsets.load(idx)),
                Int(offsets.load(idx + 1)),
            ),
        )


@export("mpg_trace")
def mpg_trace(
    hash_nodes_addr: Int,
    hash_tokens_addr: Int,
    hash_children_addr: Int,
    hash_capacity: Int,
    tokens_addr: Int,
    length: Int,
    result_addr: Int,
) abi("C") -> Int:
    var hash_nodes = IPtr(unsafe_from_address=hash_nodes_addr)
    var hash_tokens = IPtr(unsafe_from_address=hash_tokens_addr)
    var hash_children = IPtr(unsafe_from_address=hash_children_addr)
    var tokens = IPtr(unsafe_from_address=tokens_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var node = Int64(0)
    result.store(0, node)
    for pos in range(length):
        node = find_child(
            hash_nodes,
            hash_tokens,
            hash_children,
            hash_capacity,
            node,
            tokens.load(pos),
        )
        if node < 0:
            return pos + 1
        result.store(pos + 1, node)
    return length + 1


@export("mpg_bulk_find")
def mpg_bulk_find(
    hash_nodes_addr: Int,
    hash_tokens_addr: Int,
    hash_children_addr: Int,
    hash_capacity: Int,
    tokens_addr: Int,
    offsets_addr: Int,
    count: Int,
    result_addr: Int,
) abi("C"):
    var hash_nodes = IPtr(unsafe_from_address=hash_nodes_addr)
    var hash_tokens = IPtr(unsafe_from_address=hash_tokens_addr)
    var hash_children = IPtr(unsafe_from_address=hash_children_addr)
    var tokens = IPtr(unsafe_from_address=tokens_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    for idx in range(count):
        result.store(
            idx,
            find_path(
                hash_nodes,
                hash_tokens,
                hash_children,
                hash_capacity,
                tokens,
                offsets,
                idx,
            ),
        )


@export("mpg_bulk_longest")
def mpg_bulk_longest(
    hash_nodes_addr: Int,
    hash_tokens_addr: Int,
    hash_children_addr: Int,
    hash_capacity: Int,
    has_value_addr: Int,
    tokens_addr: Int,
    offsets_addr: Int,
    count: Int,
    result_addr: Int,
) abi("C"):
    var hash_nodes = IPtr(unsafe_from_address=hash_nodes_addr)
    var hash_tokens = IPtr(unsafe_from_address=hash_tokens_addr)
    var hash_children = IPtr(unsafe_from_address=hash_children_addr)
    var has_value = BPtr(unsafe_from_address=has_value_addr)
    var tokens = IPtr(unsafe_from_address=tokens_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    for idx in range(count):
        result.store(
            idx,
            longest_path(
                hash_nodes,
                hash_tokens,
                hash_children,
                hash_capacity,
                has_value,
                tokens,
                offsets,
                idx,
            ),
        )


@export("mpg_collect")
def mpg_collect(
    first_edge_addr: Int,
    edge_children_addr: Int,
    edge_next_addr: Int,
    has_value_addr: Int,
    root: Int,
    shallow: Int,
    stack_addr: Int,
    result_addr: Int,
) abi("C") -> Int:
    var first_edge = IPtr(unsafe_from_address=first_edge_addr)
    var edge_children = IPtr(unsafe_from_address=edge_children_addr)
    var edge_next = IPtr(unsafe_from_address=edge_next_addr)
    var has_value = BPtr(unsafe_from_address=has_value_addr)
    var stack = IPtr(unsafe_from_address=stack_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var node = Int64(root)
    var depth = 0
    var count = 0
    while True:
        var valued = has_value.load(Int(node)) != 0
        if valued:
            result.store(count, node)
            count += 1
        var edge = Int64(-1)
        if not (shallow != 0 and valued):
            edge = first_edge.load(Int(node))
        if edge >= 0:
            stack.store(depth, edge_next.load(Int(edge)))
            depth += 1
            node = edge_children.load(Int(edge))
            continue
        var found = False
        while depth > 0:
            depth -= 1
            edge = stack.load(depth)
            if edge >= 0:
                stack.store(depth, edge_next.load(Int(edge)))
                depth += 1
                node = edge_children.load(Int(edge))
                found = True
                break
        if not found:
            return count
