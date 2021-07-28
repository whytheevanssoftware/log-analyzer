W2V_NEIGHBORS = 20
RECURSION_LIMIT = 10**6
N_PROJ_DIM = 3
DASH_SEED = 0

pio.templates.default = "plotly_dark"
external_stylesheets_url = 'https://drive.google.com/uc?export=view&id=19OXGQ5iJIjRZD4VEZ-xiVChDmj0-SlSF'  # noqa
external_stylesheets = [external_stylesheets_url]

CACHE_CONFIG = dict()
CACHE_CONFIG['CACHE_TYPE'] = 'filesystem'
CACHE_CONFIG['CACHE_DIR'] = SOURCE + '/results/dash_cache'

def tree_parser(node, inner_list, outer_list, root_node, depth):
    d = node.key_to_child_node  # dict
    for token in list(d.keys()):
        if len(root_node.key_to_child_node.keys()) == 0:
            ret_list = []
            for row in outer_list:
                proper_len = int(row[1])
                if len(row) == proper_len+1 or len(row) + 1 == depth:
                    ret_list.append(row)
            return ret_list
        inner_list.append(token)
        child = d[token]
        if child.key_to_child_node:
            tree_parser(child, inner_list, outer_list, root_node, depth)
        else:
            d.pop(token)
            outer_list.append(inner_list)
            inner_list = ['root']
            tree_parser(root_node, inner_list, outer_list, root_node, depth)

def tree_to_list_parser(node):
    tree_df = []
    curr_path = []
    tree_dict = {}
    prev_root = [("root", node)]
    while len(prev_root) > 0:
        # Peek at last value
        curr_root = prev_root[-1]

        # Get the node element
        curr_node = curr_root[1].key_to_child_node

        # Follow path value if not already there
        if len(curr_path) <= 0 or curr_path[-1] != curr_root[0]:
            curr_path.append(curr_root[0])

        visited = False
        if curr_root[1] in tree_dict:
            visited = True
        else:
            tree_dict[curr_root[1]] = True

        # Check if value has any leaf nodes
        if not visited and len(curr_node.keys()) > 0:
            # Add those to the stack
            for nn in curr_node.items():
                prev_root.append((nn[0], nn[1]))
        else:
            # Remove previous node in the path
            prev_root.pop()

            # Record to the database if leaf
            if len(curr_node.keys()) <= 0:
                tree_df.append(deepcopy(curr_path))

            # Move back up tree
            curr_path.pop()
    return tree_df

def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2]**2)
    ptsnew[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew

def get_spherical_coords(xyz):
    sph = np.zeros(shape=xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    sph[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    sph[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    sph[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return sph

# -- Generate Data for Word Embeddings Projector -- #
def gen_scatter(embed_weights, w2v_config):
    # shape = vocab size x embedding dim size
    weights = np.ndarray(shape=(len(embed_weights), w2v_config["embed_size"]))

    # -- Populate Matrix for PCA -- #
    for idx, weight in enumerate(list(embed_weights.values())):
        weights[idx, :] = weight

    # -- Dimensionality Reduction -- #
    pca = PCA(n_components=N_PROJ_DIM, random_state=DASH_SEED).fit(weights)
    ica = FastICA(n_components=N_PROJ_DIM, random_state=DASH_SEED).fit(weights)
    srp = SparseRandomProjection(n_components=N_PROJ_DIM, random_state=DASH_SEED).fit(weights)
    reduced_embeddings = pca.transform(weights)

    # -- Calculate Nearest Neighbors -- #
    model = NearestNeighbors(n_neighbors=W2V_NEIGHBORS, algorithm='auto')
    trained_embeddings = model.fit(reduced_embeddings)

    # Currently the array has a shape of vocab size x N_PROJ_DIM and contains
    # the fitted PCA data. We need to add the vocab in the first column so
    # we know which vectors are represented.
    scatter_plot_3d_cols = ['token', 'x1', 'x2', 'x3']
    embedding_vocab_arr = np.array(list(embed_weights.keys()))
    embedding_vocab_arr = np.expand_dims(embedding_vocab_arr, 1)
    named_reduced_embeddings = np.hstack((embedding_vocab_arr, reduced_embeddings))
    scatter_plot_3d_df = pd.DataFrame(
        data=named_reduced_embeddings,
        columns=scatter_plot_3d_cols)
    scatter_plot_3d_df['x1'] = pd.to_numeric(scatter_plot_3d_df['x1'])
    scatter_plot_3d_df['x2'] = pd.to_numeric(scatter_plot_3d_df['x2'])
    scatter_plot_3d_df['x3'] = pd.to_numeric(scatter_plot_3d_df['x3'])
    
    return scatter_plot_3d_df

# By default python's recursion limit is 10**4 which is too small for our needs
def gen_treeemap(w2vp.TCL.template_miner.drain.root_node):
    sys.setrecursionlimit(RECURSION_LIMIT)

    # The root node is the master node of the tree and will be our return point
    root_node = deepcopy(w2vp.TCL.template_miner.drain.root_node)
    parsed_tree = tree_to_list_parser(root_node)
    parsed_tree_df = pd.DataFrame(data=parsed_tree)

    # The returned dataframe has generic columns so we will provide custom labels
    n_cols = len(parsed_tree_df.columns)
    col_name_list = []
    for idx in range(n_cols):
        col_name_list.append('level' + str(idx))
    parsed_tree_df.columns = col_name_list

    '''
    Without a color column our treemap would just be plain. We thought that taking
    the sum of the drain mask would be an interesting way to color the treemap.
    This lambda function will sum those values in each row and return them to a new
    columnn named 'sum'
    '''
    parsed_tree_df['sum'] = parsed_tree_df.apply(lambda x: x.str.contains('<*>'), axis=1).sum(axis=1)  # noqa
    
    return parsed_tree_df

def set_color():
    color_d = dict()
    color_d['blue'] = 'rgb(66, 133, 244)'
    color_d['red'] = 'rgb(219, 68, 55)'
    color_d['yellow'] = 'rgb(244, 180, 0)'
    color_d['orange'] = 'rgb(255, 165, 0)'
    color_d['green'] = 'rgb(15, 157, 88)'
    color_d['mint'] = 'rgb(3, 218, 198)'
    color_d['dark mint'] = 'rgb(1, 135, 134)'
    color_d['dark purple'] = 'rgb(55, 0, 179)'
    color_d['purple'] = 'rgb(98, 0, 238)'
    
    return color_d

# ================= #
#  3d Scatter Plot  #
# ================= #

    # Line formatting
def set_scatter_plot_3d_line(color_d):
    scatter_plot_3d_line = dict()
    scatter_plot_3d_line['width'] = 2
    scatter_plot_3d_line['color'] = color_d['dark mint']
    return scatter_plot_3d_line

def set_scatter_plot_3d_selected_line(color_d):
    scatter_plot_3d_selected_line = dict()
    scatter_plot_3d_selected_line['width'] = 2
    scatter_plot_3d_selected_line['color'] = color_d['dark mint']
    return scatter_plot_3d_selected_line

def set_scatter_plot_3d_nonselected_line(color_d):
    scatter_plot_3d_nonselected_line = dict()
    scatter_plot_3d_nonselected_line['width'] = 2
    scatter_plot_3d_nonselected_line['color'] = color_d['dark mint']
    return scatter_plot_3d_nonselected_line

def set_scatter_plot_3d_darker_line(color_d):
    scatter_plot_3d_darker_line = dict()
    scatter_plot_3d_darker_line['width'] = 2
    scatter_plot_3d_darker_line['color'] = color_d['dark purple']
    return scatter_plot_3d_darker_line


    # Marker formatting
def set_scatter_plot_3d_marker(color_d, scatter_plot_3d_line):
    scatter_plot_3d_marker = dict()
    scatter_plot_3d_marker['size'] = 5
    scatter_plot_3d_marker['line'] = scatter_plot_3d_line
    scatter_plot_3d_marker['color'] = color_d['mint']
    return scatter_plot_3d_marker

def set_scatter_plot_3d_selected_marker(color_d, scatter_plot_3d_selected_line):
    scatter_plot_3d_selected_marker = dict()
    scatter_plot_3d_selected_marker['size'] = 5
    scatter_plot_3d_selected_marker['color'] = color_d['mint']
    scatter_plot_3d_selected_marker['line'] = scatter_plot_3d_selected_line
    return scatter_plot_3d_selected_marker

def set_scatter_plot_3d_nonselected_marker(color_d, scatter_plot_3d_nonselected_line):
    scatter_plot_3d_nonselected_marker = dict()
    scatter_plot_3d_nonselected_marker['size'] = 5
    scatter_plot_3d_nonselected_marker['color'] = color_d['mint']
    scatter_plot_3d_nonselected_marker['opacity'] = 0.15
    scatter_plot_3d_nonselected_marker['line'] = scatter_plot_3d_nonselected_line
    return scatter_plot_3d_nonselected_line

def set_scatter_plot_3d_marker_no_color(color_d, scatter_plot_3d_darker_line):
    scatter_plot_3d_marker_no_color = dict()
    scatter_plot_3d_marker_no_color['size'] = 5
    scatter_plot_3d_marker_no_color['line'] = scatter_plot_3d_darker_line
    return scatter_plot_3d_marker_no_color

def set_scatter_plot_3d_marker_cluster_center(color_d, scatter_plot_3d_darker_line):
    scatter_plot_3d_marker_cluster_center = dict()
    scatter_plot_3d_marker_cluster_center['size'] = 10
    scatter_plot_3d_marker_cluster_center['color'] = color_d['orange']
    scatter_plot_3d_marker_cluster_center['opacity'] = 0.5
    scatter_plot_3d_marker_cluster_center['line'] = scatter_plot_3d_darker_line
    return scatter_plot_3d_marker_cluster_center

def set_scatter_plot_3d_selected_table_marker(color_d, scatter_plot_3d_darker_line):
    scatter_plot_3d_selected_table_marker = dict()
    scatter_plot_3d_selected_table_marker['size'] = 5
    scatter_plot_3d_selected_table_marker['color'] = color_d['yellow']
    scatter_plot_3d_selected_table_marker['line'] = scatter_plot_3d_darker_line
    return scatter_plot_3d_selected_table_marker


    # Style
def set_scatter_plot_3d_style():
    scatter_plot_3d_style = dict()
    scatter_plot_3d_style['height'] = '100%'
    scatter_plot_3d_style['width'] = '100%'
    return scatter_plot_3d_style


    # ========= #
    #  Treemap  #
    # ========= #

    # Style
def set_treemap_style():
    treemap_style = dict()
    treemap_style['height'] = '100%'
    treemap_style['width'] = '100%'
    return treemap_style


    # ============ #
    #  Data Table  #
    # ============ #

    # Style
def set_data_table_cell_style():
    data_table_cell_style = dict()
    data_table_cell_style['textAlign'] = 'left'
    data_table_cell_style['overflow'] = 'hidden'
    data_table_cell_style['textOverflow'] = 'ellipsis'
    data_table_cell_style['maxWidth'] = 0
    data_table_cell_style['backgroundColor'] = 'rgb(20, 20, 20)'
    data_table_cell_style['color'] = 'white'
    return data_table_cell_style

def set_data_table_header_style(color_d):
    data_table_header_style = dict()
    data_table_header_style['backgroundColor'] = color_d['purple']
    return data_table_header_style


    # ======== #
    #  Labels  #
    # ======== #

    # Style
def set_clustering_alg_drop_down_label_style():
    clustering_alg_drop_down_label_style = dict()
    clustering_alg_drop_down_label_style['color'] = 'white'
    return clustering_alg_drop_down_label_style

def set_coordinate_space_drop_down_label_style():
    coordinate_space_drop_down_label_style = dict()
    coordinate_space_drop_down_label_style['color'] = 'white'
    return coordinate_space_drop_down_label_style

def set_dim_reduction_drop_down_label_style():
    dim_reduction_drop_down_label_style = dict()
    dim_reduction_drop_down_label_style['color'] = 'white'
    return dim_reduction_drop_down_label_style

    # ================= #
    #  3d Scatter Plot  #
    # ================= #
def set_scatter_plot_3d_config():
    scatter_plot_3d_config = dict()
    scatter_plot_3d_config['responsive'] = True
    return scatter_plot_3d_config


    # ========= #
    #  Treemap  #
    # ========= #
def set_treemap_config():
    treemap_config = dict()
    treemap_config['responsive'] = True
    return treemap_config


