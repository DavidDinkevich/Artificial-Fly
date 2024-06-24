import numpy as np
import plotly
import plotly.graph_objects as go


def _disconnect_line_add_none(array1, array2):
    """
    Combines two 2D arrays such that the rows alternate between the two arrays,
    and inserts `None` (represented as `np.nan`) every two rows.

    Args:
        array1 (numpy.ndarray): The first input array with shape (n, 3).
        array2 (numpy.ndarray): The second input array with shape (n, 3).

    Returns:
        numpy.ndarray: The combined array with alternating rows from `array1` and `array2`,
        and `None` (np.nan) inserted every two rows.

    """
    combined_array = np.empty((2 * array1.shape[0], array1.shape[1]))
    combined_array[::2] = array1[:array1.shape[0]]
    combined_array[1::2] = array2[:array1.shape[0]]
    combined_array = np.insert(combined_array, range(2, combined_array.shape[0], 2), np.nan, axis=0)
    return combined_array


def _scatter_3d(fig, data, hover_data, name, **kwargs):
    """3d plotly scatter

    Args:
        fig (plotly): plotly figure
        data (np array): contins x,y,z (axis per column)
        hover_data (_type_): data to show when hoverinb, such as time
        name (string): name of poins

    Returns:
        fig (plotly): plotly figure
    """

    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        customdata=hover_data.astype(int), **kwargs,
        name=name,
    ))
    fig.update_traces(hovertemplate='<b>time</b>: %{customdata:,.f}<br>')
    return fig


def _plot_3d_traj(data, plot_cofnig, mov_name, exp_name, color_prop):
    """plot 3d trajectory

    Args:
        data (dict): dictionary contining cm, time, x_vector, y_vector, start_pert_endpert, color_prop
        plot_cofnig (dict):fly_samples - delta sample of the fly axes .
                                    traj_samples - delta samle of cm
                                    size_x - scale of body axis
                                    size_y - scale of y axis
                                    delta_y_on_x - location of y axis on x axis (make it look like a cross)

        mov_name (str): name of movie
        exp_name (str): name of experiment
        color_prop (str): name of property to color cm

    Returns:
        fig
    """
    fig = go.Figure()

    scene = dict(camera=dict(eye=dict(x=1., y=1, z=1.25)),
                 xaxis=dict(nticks=10),
                 yaxis=dict(nticks=10),
                 zaxis=dict(nticks=10),
                 aspectmode='data'
                 )

    fig.update_scenes(scene)

    vector_kwargs = {'marker': dict(size=5, opacity=0.8, color='red'), 'line': dict(color='black', width=8)}
    cm_kwargs = {'marker': dict(size=8, opacity=0.8, color=data[color_prop][::plot_cofnig['traj_samples']],
                                colorbar=dict(thickness=10, title=color_prop), colorscale='jet', colorbar_x=-0.1),
                 'mode': 'markers'}
    pertubations = {'marker': dict(size=13, opacity=0, color=['pink', 'green', 'red']), 'mode': 'markers'}

    start_pert_endpert = data['start_pert_endpert']

    fig = _scatter_3d(fig, data['cm'][::plot_cofnig['traj_samples'], :],
                     data['time'][::plot_cofnig['traj_samples']], name='CM', **cm_kwargs)

    fig = _scatter_3d(fig, data['x_vector'], data['time'][::plot_cofnig['fly_samples']], name='X vector',
                      **vector_kwargs)
    fig = _scatter_3d(fig, data['y_vector'], data['time'][::plot_cofnig['fly_samples']], name='Y vector',
                      **vector_kwargs)

    if 'forces' in data:
        forces_kwargs = {'marker': dict(size=5, opacity=0.8, color='blue'), 'line': dict(color='blue', width=5)}
        fig = _scatter_3d(fig, data['forces'], data['time'][::plot_cofnig['fly_samples']], name='forces',
                          **forces_kwargs)
    fig = _scatter_3d(fig, data['cm'][start_pert_endpert, :], data['time'][start_pert_endpert], name='pertubations',
                      **pertubations)
    fig.update_layout(title=f'{mov_name} {exp_name}', coloraxis_colorbar_y=0.5, coloraxis_colorscale='jet')
    return fig


def _prepare_data(input_data, color_prop, plot_cofnig, pertubation):
    """Calulations for ploting a 3d trajectory

    Args:
        color_prop (str, optional):property to color the cm .
        plot_cofnig (dict, optional):fly_samples - delta sample of the fly axes .
                                    traj_samples - delta samle of cm
                                    size_x - scale of body axis
                                    size_y - scale of y axis
                                    delta_y_on_x - location of y axis on x axis (make it look like a cross)

    Returns:
        data (dict): a dictionary with all relevant data
        plot_cofig (dict) : the configuration of the plot
    """
    data = {}
    data[color_prop] = input_data[color_prop]
    data['cm'] = input_data[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy() * 1000
    data['time'] = input_data['t_ms'].to_numpy()
    vectors = {ax: input_data[[f'{ax}_body_x', f'{ax}_body_y', f'{ax}_body_z']].to_numpy() for ax in ['x', 'y']}

    body_x_vector = vectors['x'] * plot_cofnig['size_x'] + data['cm']
    body_y_vector = vectors['y'][::plot_cofnig['fly_samples'], :] * plot_cofnig['size_y']
    delta_y_on_x = (vectors['x'][::plot_cofnig['fly_samples'], :]) * plot_cofnig['delta_y_on_x'] + data['cm'][
                                                                                                   ::plot_cofnig[
                                                                                                       'fly_samples'],
                                                                                                   :]  # define the size of Ybody vecotr

    # disconnect the coordinates to get line for the vectors
    data['x_vector'] = _disconnect_line_add_none(data['cm'][::plot_cofnig['fly_samples'], :],
                                                body_x_vector[::plot_cofnig['fly_samples'], :])
    data['y_vector'] = _disconnect_line_add_none(-body_y_vector + delta_y_on_x, body_y_vector + delta_y_on_x)

    # index for pertubationmarker
    idx_end_pertubation = np.where((data['time'] < (pertubation + 1)) & (data['time'] > (pertubation - 1)))[0][
        0] if pertubation != False else False
    idx_start_pertubation = np.where(data['time'] == 0)[0][0]
    data['start_pert_endpert'] = [0, idx_start_pertubation, idx_end_pertubation] if pertubation != False else [0,
                                                                                                               idx_start_pertubation]
    return data


def plot_3d_trajectory_in_browser(pandas_frame):
    plot_config = {'fly_samples': 150, 'traj_samples': 20, 'size_x': 1, 'size_y': 1 / 3, 'delta_y_on_x': 3 / 4}
    data = _prepare_data(pandas_frame, 'Beats per Sec', plot_config, False)
    fig = _plot_3d_traj(data, plot_config, "Trajectory", "1", 'Beats per Sec')
    fig.show()


def save_3d_trajectory(pandas_frame, path):
    plot_config = {'fly_samples': 150, 'traj_samples': 20, 'size_x': 1, 'size_y': 1 / 3, 'delta_y_on_x': 3 / 4}
    data = _prepare_data(pandas_frame, 'Beats per Sec', plot_config, False)
    fig = _plot_3d_traj(data, plot_config, "Trajectory", "1", 'Beats per Sec')
    plotly.offline.plot(fig, filename=f'{path}.html', auto_open=False)
