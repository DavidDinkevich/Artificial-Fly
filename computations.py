
import numpy as np
from numpy import sin, cos, tanh, arcsin
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation


DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi


def body_ang_vel_pqr(angles, angles_dot, get_pqr):
    """
    Converts change in euler angles to body rates (if get_pqr is True) or body rates to euler rates (if get_pqr is False)
    :param angles: euler angles (np.array[psi,theta,phi])
    :param angles_dot: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    :param get_pqr: whether to get body rates from euler rates or the other way around
    :return: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    """
    psi = angles[0]
    theta = angles[1]
    psi_p_dot = angles_dot[0]
    theta_q_dot = angles_dot[1]
    phi_r_dot = angles_dot[2]

    if get_pqr:
        p = psi_p_dot - phi_r_dot * sin(theta)
        q = theta_q_dot * cos(psi) + phi_r_dot * sin(psi) * cos(theta)
        r = -theta_q_dot * sin(psi) + phi_r_dot * cos(psi) * cos(theta)
        return np.array([p, q, r])
    else:
        psi_dot = psi_p_dot + theta_q_dot * (sin(psi) * sin(theta)) / cos(theta) + phi_r_dot * (
                cos(psi) * sin(theta)) / cos(theta)
        theta_dot = theta_q_dot * cos(psi) + phi_r_dot * -sin(psi)
        phi_dot = theta_q_dot * sin(psi) / cos(theta) + phi_r_dot * cos(psi) / cos(theta)
        return np.array([psi_dot, theta_dot, phi_dot])


def wing_angles(psi, theta, phi, omega, delta_psi, delta_theta, c, k, t):
    """
    Computes the wing angles given a set of variables, described in (see Whitehead et al, "Pitch perfect: how fruit flies
    control their body pitch angle." 2015, appendix 1)
    :param psi: [psi0_L psim_L psi0_R psim_R].psi0_R = 90, psim_R = -psim_L;
    :param theta: [theta0_L thetam_L theta0_R thetam_R].theta0_R = theta0_L, thetam_R =Wing.thetam_L[rad]
    :param phi: [phi0_L phim_L phi0_R phim_R].phi0_R = -phi0_L, phim_R = -phim_L[rad]
    :param omega: wing angular velocity [rad / s]
    :param delta_psi: wing angles phase [rad]
    :param delta_theta: wing angles phase [rad]
    :param c:
    :param k:
    :param t: time in cycle
    :return:
    """

    psi_w = psi[0] + psi[1] * tanh(c * np.sin(omega * t + delta_psi)) / tanh(c)
    theta_w = theta[0] + theta[1] * np.cos(2 * omega * t + delta_theta)
    phi_w = phi[0] + phi[1] * arcsin(k * sin(omega * t)) / arcsin(k)

    psi_dot = -(c * omega * psi[1] * np.cos(delta_psi + omega * t) * (
            np.tanh(c * (sin(delta_psi + omega * t))) ** 2 - 1)) / tanh(c)
    theta_dot = -2 * omega * theta[1] * sin(delta_theta + 2 * omega * t)
    phi_dot = k * (omega * phi[1] * cos(omega * t)) / (
            arcsin(k) * (1 - k ** 2 * (sin(omega * t)) ** 2) ** (1 / 2))

    angles = np.array([psi_w, theta_w, phi_w])
    angles_dot = np.array([psi_dot, theta_dot, phi_dot])
    return angles, angles_dot


def wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, wing_rl, t, fly_env):
    '''
    Computation copied from the matlab version of this simulation
    '''

    angles, angles_dot = wing_angles(u1, u2, u3, fly_env.wing_freq, fly_env.config['wing']['delta_psi'],
                                     fly_env.config['wing']['delta_theta'], fly_env.config['wing']['C'], fly_env.config['wing']['K'], t)
    r_wing2lab = Rotation.from_euler('xyz', [angles[0], angles[1], angles[2]]).as_matrix()
    r_sp2lab = Rotation.from_euler('xyz', fly_env.config['gen']['strkplnAng']).as_matrix()
    r_body2lab = Rotation.from_euler('xyz', [x7, x8, x9]).as_matrix()
    r_spwithbod2lab = r_body2lab @ r_sp2lab

    # Wing velocity
    angular_vel = body_ang_vel_pqr(angles, angles_dot, True)
    tang_wing_v = np.cross(angular_vel, fly_env.config['wing']['speedCalc'])

    # Body velocity
    ac_lab = r_spwithbod2lab @ r_wing2lab @ fly_env.config['wing']['speedCalc']
    ac_bod = r_body2lab.T @ ac_lab
    vel_loc_bod = ac_bod + fly_env.config['wing'][f'hinge{wing_rl}'].T
    vb = np.cross(np.array([x4, x5, x6]), vel_loc_bod) + np.array([x1, x2, x3])
    vb = r_body2lab @ vb
    vb_lab = r_spwithbod2lab.T @ vb
    vb_wing = r_wing2lab.T @ vb_lab
    vw = vb_wing + tang_wing_v.T
    alpha = np.arctan2(vw[2], vw[1])
    if wing_rl == 'R':
        cl, cd, span_hat, lhat, drag, lift, t_body, f_body_aero, f_lab_aero, t_lab = fly_env.wing_model_right.get_forces(
            alpha, vw,
            r_body2lab,
            r_wing2lab,
            r_spwithbod2lab)
    if wing_rl == 'L':
        cl, cd, span_hat, lhat, drag, lift, t_body, f_body_aero, f_lab_aero, t_lab = fly_env.wing_model_left.get_forces(
            alpha, vw,
            r_body2lab,
            r_wing2lab,
            r_spwithbod2lab)
    return (f_body_aero, t_body, r_body2lab, r_wing2lab, r_spwithbod2lab, cl, cd, angles.T,
            span_hat, lhat, drag, lift, f_lab_aero, t_lab, ac_lab)


def fly_solve_diff(t, y, fly_env):
    '''
    Computation copied from the matlab version of this simulation
    '''
    
    x1 = y[0]  # u
    x2 = y[1]  # v
    x3 = y[2]  # w
    x4 = y[3]  # p
    x5 = y[4]  # q
    x6 = y[5]  # r
    x7 = y[6]  # psi(roll)
    x8 = y[7]  # theta(pitch)
    x9 = y[8]  # phi(roll)

    u1 = fly_env.curr_wing_state_left.psi
    u2 = fly_env.curr_wing_state_left.theta
    u3 = fly_env.curr_wing_state_left.phi
    u4 = fly_env.curr_wing_state_right.psi
    u5 = fly_env.curr_wing_state_right.theta
    u6 = fly_env.curr_wing_state_right.phi

    wingout_r = wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u4, u5, u6, 'R', t, fly_env)
    wingout_l = wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, 'L', t, fly_env)
    vb = np.array([x1, x2, x3])
    fb = wingout_r[0] + wingout_l[0] + fly_env.config['gen']['m'] * wingout_r[2].T @ fly_env.config['gen']['g'].T
    tb = wingout_r[1] + wingout_l[1]

    omega_b = np.array([x4, x5, x6]).T
    x1to3dot = (1 / fly_env.config['gen']['m']) * fb - np.cross(omega_b, vb)
    x4to6dot = 1000*inv(1000*fly_env.config['gen']['I']) @ (tb - np.cross(omega_b, fly_env.config['gen']['I'] @ omega_b))
    x7to9dot = body_ang_vel_pqr(np.array([x7, x8, x9]), omega_b, False)
    y_dot = np.concatenate([x1to3dot, x4to6dot, x7to9dot])
    return y_dot, wingout_l, wingout_r


class AeroModel(object):
    def __init__(self, span, chord, rho, r22, clmax, cdmax, cd0, hinge_loc, ac_loc):
        self.s = span * chord * np.pi / 4
        self.rho = rho
        self.r22 = r22
        self.clmax = clmax
        self.cdmax = cdmax
        self.cd0 = cd0
        self.span_hat = np.array([1, 0, 0])
        self.hinge_location = hinge_loc
        self.ac_loc = ac_loc

    def get_forces(self, aoa, v_wing, rotation_mat_body2lab, rotation_mat_wing2lab, rotation_mat_sp2lab):
        '''
        Computation copied from the matlab version of this simulation
        '''

        cl = self.clmax * sin(2 * aoa)
        cd = (self.cdmax + self.cd0) / 2 - (self.cdmax - self.cd0) / 2 * cos(2 * aoa)
        u = v_wing[0] ** 2 + v_wing[1] ** 2 + v_wing[2] ** 2
        uhat = v_wing / norm(v_wing)
        span_hat = self.span_hat
        lhat = (np.cross(span_hat, -uhat)).T  # perpendicular to Uhat
        lhat = lhat / norm(lhat)
        q = self.rho * self.s * self.r22 * u
        drag = -0.5 * cd * q * uhat
        lift = 0.5 * cl * q * lhat
        rot_mat_spw2lab = rotation_mat_sp2lab @ rotation_mat_wing2lab
        ac_loc_lab = rot_mat_spw2lab @ self.ac_loc.T + rotation_mat_body2lab @ self.hinge_location.T  # AC location in lab axes
        ac_loc_body = rotation_mat_body2lab.T @ ac_loc_lab  # AC location in body axes

        f_lab_aero = rot_mat_spw2lab @ lift + rot_mat_spw2lab @ drag
        # force in body axes
        f_body = rotation_mat_body2lab.T @ f_lab_aero
        t_lab = np.cross(ac_loc_lab.T,
                         f_lab_aero).T  # + cross(ACLocB_body, Dbod).T # torque on body (in body axes)
        # (from forces, no CM0)
        t_body = np.cross(ac_loc_body.T,
                          f_body).T  # + cross(ACLocB_lab, Dbod_lab) # torque on body( in bodyaxes)
        # (from forces, no CM0)
        return cl, cd, span_hat, lhat, drag, lift, t_body, f_body, f_lab_aero, t_lab
