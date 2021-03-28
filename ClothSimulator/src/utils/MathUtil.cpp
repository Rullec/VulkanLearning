#include "MathUtil.h"
#include "LogUtil.h"
#include <iostream>
#include <time.h>
// const enum eRotationOrder gRotationOrder = eRotationOrder::XYZ;
const tVector gGravity = tVector(0, -9.8, 0, 0);
// const tVector gGravity = tVector(0, 0, 0, 0);
cRand cMathUtil::gRand = cRand();

int cMathUtil::Clamp(int val, int min, int max)
{
    return std::max(min, std::min(val, max));
}

void cMathUtil::Clamp(const Eigen::VectorXd &min, const Eigen::VectorXd &max,
                      Eigen::VectorXd &out_vec)
{
    out_vec = out_vec.cwiseMin(max).cwiseMax(min);
}

double cMathUtil::Clamp(double val, double min, double max)
{
    return std::max(min, std::min(val, max));
}

double cMathUtil::Saturate(double val) { return Clamp(val, 0.0, 1.0); }

double cMathUtil::Lerp(double t, double val0, double val1)
{
    return (1 - t) * val0 + t * val1;
}

double cMathUtil::NormalizeAngle(double theta)
{
    // normalizes theta to be between [-pi, pi]
    double norm_theta = fmod(theta, 2 * M_PI);
    if (norm_theta > M_PI)
    {
        norm_theta = -2 * M_PI + norm_theta;
    }
    else if (norm_theta < -M_PI)
    {
        norm_theta = 2 * M_PI + norm_theta;
    }
    return norm_theta;
}

double cMathUtil::RandDouble() { return RandDouble(0, 1); }

double cMathUtil::RandDouble(double min, double max)
{
    return gRand.RandDouble(min, max);
}

double cMathUtil::RandDoubleNorm(double mean, double stdev)
{
    return gRand.RandDoubleNorm(mean, stdev);
}

double cMathUtil::RandDoubleExp(double lambda)
{
    return gRand.RandDoubleExp(lambda);
}

double cMathUtil::RandDoubleSeed(double seed)
{
    unsigned int int_seed = *reinterpret_cast<unsigned int *>(&seed);
    std::default_random_engine rand_gen(int_seed);
    std::uniform_real_distribution<double> dist;
    return dist(rand_gen);
}

int cMathUtil::RandInt() { return gRand.RandInt(); }

int cMathUtil::RandInt(int min, int max) { return gRand.RandInt(min, max); }

int cMathUtil::RandUint() { return gRand.RandUint(); }

int cMathUtil::RandUint(unsigned int min, unsigned int max)
{
    return gRand.RandUint(min, max);
}

int cMathUtil::RandIntExclude(int min, int max, int exc)
{
    return gRand.RandIntExclude(min, max, exc);
}

void cMathUtil::SeedRand(unsigned long int seed)
{
    gRand.Seed(seed);
    srand(gRand.RandInt());
}

int cMathUtil::RandSign() { return gRand.RandSign(); }

double cMathUtil::SmoothStep(double t)
{
    double val = t * t * t * (t * (t * 6 - 15) + 10);
    return val;
}

bool cMathUtil::FlipCoin(double p) { return gRand.FlipCoin(p); }

tMatrix cMathUtil::TranslateMat(const tVector &trans)
{
    tMatrix mat = tMatrix::Identity();
    mat(0, 3) = trans[0];
    mat(1, 3) = trans[1];
    mat(2, 3) = trans[2];
    return mat;
}

tMatrix cMathUtil::ScaleMat(double scale)
{
    return ScaleMat(tVector::Ones() * scale);
}

tMatrix cMathUtil::ScaleMat(const tVector &scale)
{
    tMatrix mat = tMatrix::Identity();
    mat(0, 0) = scale[0];
    mat(1, 1) = scale[1];
    mat(2, 2) = scale[2];
    return mat;
}

tMatrix cMathUtil::RotateMat(const tVector &euler,
                             const eRotationOrder gRotationOrder)
{
    double x = euler[0];
    double y = euler[1];
    double z = euler[2];

    double sinx = std::sin(x);
    double cosx = std::cos(x);
    double siny = std::sin(y);
    double cosy = std::cos(y);
    double sinz = std::sin(z);
    double cosz = std::cos(z);

    tMatrix mat = tMatrix::Identity();

    if (gRotationOrder == eRotationOrder::XYZ)
    {
        mat(0, 0) = cosy * cosz;
        mat(1, 0) = cosy * sinz;
        mat(2, 0) = -siny;

        mat(0, 1) = sinx * siny * cosz - cosx * sinz;
        mat(1, 1) = sinx * siny * sinz + cosx * cosz;
        mat(2, 1) = sinx * cosy;

        mat(0, 2) = cosx * siny * cosz + sinx * sinz;
        mat(1, 2) = cosx * siny * sinz - sinx * cosz;
        mat(2, 2) = cosx * cosy;
    }
    else
    {
        std::cout << "[error] cMathUtil::RotateMat(const tVector& euler): "
                     "Unsupported rotation order"
                  << std::endl;
        exit(1);
    }
    return mat;
}

tMatrix cMathUtil::RotateMat(const tVector &axis, double theta)
{
    assert(std::abs(axis.squaredNorm() - 1) < 0.0001);
    double c = std::cos(theta);
    double s = std::sin(theta);
    double x = axis[0];
    double y = axis[1];
    double z = axis[2];

    tMatrix mat;
    mat << c + x * x * (1 - c), x * y * (1 - c) - z * s,
        x * z * (1 - c) + y * s, 0, y * x * (1 - c) + z * s,
        c + y * y * (1 - c), y * z * (1 - c) - x * s, 0,
        z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c),
        0, 0, 0, 0, 1;

    return mat;
}

tMatrix cMathUtil::RotateMat(const tQuaternion &q)
{
    tMatrix mat = tMatrix::Identity();

    double sqw = q.w() * q.w();
    double sqx = q.x() * q.x();
    double sqy = q.y() * q.y();
    double sqz = q.z() * q.z();
    double invs = 1 / (sqx + sqy + sqz + sqw);

    mat(0, 0) = (sqx - sqy - sqz + sqw) * invs;
    mat(1, 1) = (-sqx + sqy - sqz + sqw) * invs;
    mat(2, 2) = (-sqx - sqy + sqz + sqw) * invs;

    double tmp1 = q.x() * q.y();
    double tmp2 = q.z() * q.w();
    mat(1, 0) = 2.0 * (tmp1 + tmp2) * invs;
    mat(0, 1) = 2.0 * (tmp1 - tmp2) * invs;

    tmp1 = q.x() * q.z();
    tmp2 = q.y() * q.w();
    mat(2, 0) = 2.0 * (tmp1 - tmp2) * invs;
    mat(0, 2) = 2.0 * (tmp1 + tmp2) * invs;

    tmp1 = q.y() * q.z();
    tmp2 = q.x() * q.w();
    mat(2, 1) = 2.0 * (tmp1 + tmp2) * invs;
    mat(1, 2) = 2.0 * (tmp1 - tmp2) * invs;
    return mat;
}

tMatrix cMathUtil::CrossMat(const tVector &a)
{
    tMatrix m;
    m << 0, -a[2], a[1], 0, a[2], 0, -a[0], 0, -a[1], a[0], 0, 0, 0, 0, 0, 1;
    return m;
}

tMatrix cMathUtil::InvRigidMat(const tMatrix &mat)
{
    tMatrix inv_mat = tMatrix::Zero();
    inv_mat.block(0, 0, 3, 3) = mat.block(0, 0, 3, 3).transpose();
    inv_mat.col(3) = -inv_mat * mat.col(3);
    inv_mat(3, 3) = 1;
    return inv_mat;
}

tVector cMathUtil::GetRigidTrans(const tMatrix &mat)
{
    return tVector(mat(0, 3), mat(1, 3), mat(2, 3), 0);
}

tVector cMathUtil::InvEuler(const tVector &euler,
                            const eRotationOrder gRotationOrder)
{
    if (gRotationOrder == eRotationOrder::XYZ)
    {
        tMatrix inv_mat = cMathUtil::RotateMat(tVector(1, 0, 0, 0), -euler[0]) *
                          cMathUtil::RotateMat(tVector(0, 1, 0, 0), -euler[1]) *
                          cMathUtil::RotateMat(tVector(0, 0, 1, 0), -euler[2]);
        tVector inv_euler =
            cMathUtil::RotMatToEuler(inv_mat, eRotationOrder::XYZ);
        return inv_euler;
    }
    else
    {
        std::cout << "[error] cMathUtil::InvEuler: Unsupported rotation order"
                  << std::endl;
        exit(1);
    }
}

void cMathUtil::RotMatToAxisAngle(const tMatrix &mat, tVector &out_axis,
                                  double &out_theta)
{
    double c = (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1) * 0.5;
    c = cMathUtil::Clamp(c, -1.0, 1.0);

    out_theta = std::acos(c);
    if (std::abs(out_theta) < 0.00001)
    {
        out_axis = tVector(0, 0, 1, 0);
    }
    else
    {
        double m21 = mat(2, 1) - mat(1, 2);
        double m02 = mat(0, 2) - mat(2, 0);
        double m10 = mat(1, 0) - mat(0, 1);
        double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
        out_axis[0] = m21 / denom;
        out_axis[1] = m02 / denom;
        out_axis[2] = m10 / denom;
        out_axis[3] = 0;
    }
}

tVector cMathUtil::RotMatToEuler(const tMatrix &mat,
                                 const eRotationOrder gRotationOrder)
{
    tVector euler;
    if (gRotationOrder == eRotationOrder::XYZ)
    {

        euler[0] = std::atan2(mat(2, 1), mat(2, 2));
        euler[1] = std::atan2(-mat(2, 0), std::sqrt(mat(2, 1) * mat(2, 1) +
                                                    mat(2, 2) * mat(2, 2)));
        euler[2] = std::atan2(mat(1, 0), mat(0, 0));
        euler[3] = 0;
    }
    else
    {
        std::cout << "[error] cMathUtil::RotateMat: Unsupported rotation order"
                  << std::endl;
        exit(1);
    }

    return euler;
}

tMatrix cMathUtil::AxisAngleToRotmat(const tVector &angvel)
{
    return cMathUtil::RotMat(AxisAngleToQuaternion(angvel));
}

tVector cMathUtil::EulerangleToAxisAngle(const tVector &euler,
                                         const eRotationOrder gRotationOrder)
{
    tVector axis = tVector::Zero();
    double angle = 0;
    cMathUtil::EulerToAxisAngle(euler, axis, angle, gRotationOrder);
    return axis * angle;
}
tQuaternion cMathUtil::RotMatToQuaternion(const tMatrix &mat)
{
    double tr = mat(0, 0) + mat(1, 1) + mat(2, 2);
    tQuaternion q;
    if (tr > 0)
    {
        double S = sqrt(tr + 1.0) * 2; // S=4*qw
        q.w() = 0.25 * S;
        q.x() = (mat(2, 1) - mat(1, 2)) / S;
        q.y() = (mat(0, 2) - mat(2, 0)) / S;
        q.z() = (mat(1, 0) - mat(0, 1)) / S;
    }
    else if ((mat(0, 0) > mat(1, 1) && (mat(0, 0) > mat(2, 2))))
    {
        double S = sqrt(1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2)) * 2; // S=4*qx
        q.w() = (mat(2, 1) - mat(1, 2)) / S;
        q.x() = 0.25 * S;
        q.y() = (mat(0, 1) + mat(1, 0)) / S;
        q.z() = (mat(0, 2) + mat(2, 0)) / S;
    }
    else if (mat(1, 1) > mat(2, 2))
    {
        double S = sqrt(1.0 + mat(1, 1) - mat(0, 0) - mat(2, 2)) * 2; // S=4*qy
        q.w() = (mat(0, 2) - mat(2, 0)) / S;
        q.x() = (mat(0, 1) + mat(1, 0)) / S;
        q.y() = 0.25 * S;
        q.z() = (mat(1, 2) + mat(2, 1)) / S;
    }
    else
    {
        double S = sqrt(1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1)) * 2; // S=4*qz
        q.w() = (mat(1, 0) - mat(0, 1)) / S;
        q.x() = (mat(0, 2) + mat(2, 0)) / S;
        q.y() = (mat(1, 2) + mat(2, 1)) / S;
        q.z() = 0.25 * S;
    }

    return q;
}

void cMathUtil::EulerToAxisAngle(const tVector &euler, tVector &out_axis,
                                 double &out_theta,
                                 const eRotationOrder gRotationOrder)
{

    if (gRotationOrder == eRotationOrder::XYZ)
    {
        double x = euler[0];
        double y = euler[1];
        double z = euler[2];

        double sinx = std::sin(x);
        double cosx = std::cos(x);
        double siny = std::sin(y);
        double cosy = std::cos(y);
        double sinz = std::sin(z);
        double cosz = std::cos(z);

        double c =
            (cosy * cosz + sinx * siny * sinz + cosx * cosz + cosx * cosy - 1) *
            0.5;
        c = Clamp(c, -1.0, 1.0);

        out_theta = std::acos(c);
        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = sinx * cosy - cosx * siny * sinz + sinx * cosz;
            double m02 = cosx * siny * cosz + sinx * sinz + siny;
            double m10 = cosy * sinz - sinx * siny * cosz + cosx * sinz;
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    else
    {
        std::cout << "[error] cMathUtil::EulerToAxisAngle: Unsupported "
                     "rotation order"
                  << std::endl;
        exit(1);
    }
}

tVector cMathUtil::AxisAngleToEuler(const tVector &axis, double theta)
{
    tQuaternion q = AxisAngleToQuaternion(axis, theta);
    return QuaternionToEuler(q, eRotationOrder::XYZ);
}

tMatrix cMathUtil::DirToRotMat(const tVector &dir, const tVector &up)
{
    tVector x = up.cross3(dir);
    double x_norm = x.norm();
    if (x_norm == 0)
    {
        x_norm = 1;
        x = (dir.dot(up) >= 0) ? tVector(1, 0, 0, 0) : tVector(-1, 0, 0, 0);
    }
    x /= x_norm;

    tVector y = dir.cross3(x).normalized();
    tVector z = dir;

    tMatrix mat = tMatrix::Identity();

    mat.block(0, 0, 3, 1) = x.segment(0, 3);
    mat.block(0, 1, 3, 1) = y.segment(0, 3);
    mat.block(0, 2, 3, 1) = z.segment(0, 3);
    return mat;
}

void cMathUtil::DeltaRot(const tVector &axis0, double theta0,
                         const tVector &axis1, double theta1, tVector &out_axis,
                         double &out_theta)
{
    tMatrix R0 = RotateMat(axis0, theta0);
    tMatrix R1 = RotateMat(axis1, theta1);
    tMatrix M = DeltaRot(R0, R1);
    RotMatToAxisAngle(M, out_axis, out_theta);
}

tMatrix cMathUtil::DeltaRot(const tMatrix &R0, const tMatrix &R1)
{
    return R1 * R0.transpose();
}

tQuaternion cMathUtil::EulerToQuaternion(const tVector &euler,
                                         const eRotationOrder order)
{
    tVector axis;
    double theta;
    EulerToAxisAngle(euler, axis, theta, order);
    return AxisAngleToQuaternion(axis, theta);
}

tQuaternion cMathUtil::CoefVectorToQuaternion(const tVector &coef)
{
    // coef = [x, y, z, w]
    return tQuaternion(coef[3], coef[0], coef[1], coef[2]);
}

tVector cMathUtil::QuaternionToEuler(const tQuaternion &q,
                                     const eRotationOrder gRotationOrder)
{
    if (gRotationOrder == eRotationOrder::XYZ)
    {
        double sinr = 2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        double x = std::atan2(sinr, cosr);

        double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
        double y = 0;
        if (fabs(sinp) >= 1) // north pole and south pole
        {
            y = copysign(M_PI / 2,
                         sinp); // use 90 degrees if out of range
        }
        else
        {
            y = asin(sinp);
        }

        double siny = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        double z = std::atan2(siny, cosy);

        return tVector(x, y, z, 0);
    }
    else
    {
        std::cout << "[error] cMathUtil::QuaternionToEuler: Unsupported "
                     "rotation order"
                  << std::endl;
        exit(1);
    }
}

tQuaternion cMathUtil::AxisAngleToQuaternion(const tVector &axis, double theta)
{
    // axis must be normalized
    // std::cout << axis.transpose() << std::endl;
    SIM_ASSERT(std::fabs(axis.norm() - 1) < 1e-10 ||
               std::fabs(axis.norm()) < 1e-10);
    double c = std::cos(theta / 2);
    double s = std::sin(theta / 2);
    tQuaternion q;
    q.w() = c;
    q.x() = s * axis[0];
    q.y() = s * axis[1];
    q.z() = s * axis[2];
    if (q.w() < 0)
        q = cMathUtil::MinusQuaternion(q);
    return q;
}
tVector cMathUtil::QuaternionToAxisAngle(const tQuaternion &q)
{
    tVector out_axis;
    double out_theta;
    QuaternionToAxisAngle(q, out_axis, out_theta);

    out_axis *= out_theta;
    out_axis[3] = 0;
    return out_axis;
}

void cMathUtil::QuaternionToAxisAngle(const tQuaternion &q, tVector &out_axis,
                                      double &out_theta)
{
    out_theta = 0;
    out_axis = tVector(0, 0, 1, 0);

    tQuaternion q1 = q;
    if (q1.w() > 1)
    {
        q1.normalize();
    }

    double sin_theta = std::sqrt(1 - q1.w() * q1.w());
    if (sin_theta > 0.000001)
    {
        out_theta = 2 * std::acos(q1.w());
        out_theta = cMathUtil::NormalizeAngle(out_theta);
        out_axis = tVector(q1.x(), q1.y(), q1.z(), 0) / sin_theta;
    }
}

tMatrix cMathUtil::BuildQuaternionDiffMat(const tQuaternion &q)
{
    // it's right
    tMatrix mat;
    mat << -0.5 * q.x(), -0.5 * q.y(), -0.5 * q.z(), 0, // for w
        0.5 * q.w(), -0.5 * q.z(), 0.5 * q.y(), 0,      // for x
        0.5 * q.z(), 0.5 * q.w(), -0.5 * q.x(), 0,      // for y
        -0.5 * q.y(), 0.5 * q.x(), 0.5 * q.w(), 0;      // for z
    return mat;
}

tVector cMathUtil::CalcQuaternionVel(const tQuaternion &q0,
                                     const tQuaternion &q1, double dt)
{
    tQuaternion q_diff = cMathUtil::QuatDiff(q0, q1);
    tVector axis;
    double theta;
    QuaternionToAxisAngle(q_diff, axis, theta);
    return (theta / dt) * axis;
}

tVector cMathUtil::CalcQuaternionVelRel(const tQuaternion &q0,
                                        const tQuaternion &q1, double dt)
{
    // calculate relative rotational velocity in the coordinate frame of q0
    tQuaternion q_diff = q0.conjugate() * q1;
    tVector axis;
    double theta;
    QuaternionToAxisAngle(q_diff, axis, theta);
    return (theta / dt) * axis;
}

tQuaternion cMathUtil::VecToQuat(const tVector &v)
{
    // v format: [w, x, y, z]
    return tQuaternion(v[0], v[1], v[2], v[3]);
}

tVector cMathUtil::QuatToVec(const tQuaternion &q)
{
    // return value format : [w, x, y, z]
    return tVector(q.w(), q.x(), q.y(), q.z());
}

tQuaternion cMathUtil::QuatDiff(const tQuaternion &q0, const tQuaternion &q1)
{
    return q1 * q0.conjugate();
}

double cMathUtil::QuatDiffTheta(const tQuaternion &q0, const tQuaternion &q1)
{
    tQuaternion dq = QuatDiff(q0, q1);
    return QuatTheta(dq);
}

// given a
double cMathUtil::QuatTheta(const tQuaternion &dq)
{
    double theta = 0;
    tQuaternion q1 = dq;
    if (q1.w() > 1)
    {
        q1.normalize();
    }

    // theta = angle / 2
    double sin_theta = std::sqrt(
        1 -
        q1.w() *
            q1.w()); // sin(theta) which "theta" is the rotation angle/2 in dq
    if (sin_theta > 1e-7)
    {
        theta = 2 * std::acos(q1.w());            // this is angle now
        theta = cMathUtil::NormalizeAngle(theta); // noramlize angle
    }
    return theta;
}

/**
 * \brief               Calculate d(q1 * q0.conj) / dq0
*/
tMatrix cMathUtil::Calc_Dq1q0conj_Dq0(const tQuaternion &q0,
                                      const tQuaternion &q1)
{
    double a1 = q1.w(), b1 = q1.x(), c1 = q1.y(), d1 = q1.z();
    tMatrix deriv = tMatrix::Zero();
    deriv.col(0) = tVector(a1, b1, c1, d1);
    deriv.col(1) = tVector(b1, -a1, -d1, c1);
    deriv.col(2) = tVector(c1, d1, -a1, -b1);
    deriv.col(3) = tVector(d1, -c1, b1, -a1);
    return deriv;
}

/**
 * \brief           calculate d(Quaternion)/(daxis angle)
*/
tMatrix cMathUtil::Calc_DQuaternion_DAxisAngle(const tVector &aa)
{
    double theta = aa.norm();
    tMatrix dQuaterniondAA = tMatrix::Zero();

    if (std::fabs(theta) < 1e-5)
    {
        dQuaterniondAA.row(0) = -1 / 3 * aa.transpose();
        dQuaterniondAA(1, 0) = 0.5;
        dQuaterniondAA(2, 1) = 0.5;
        dQuaterniondAA(3, 2) = 0.5;
    }
    else
    {
        dQuaterniondAA.row(0) =
            -0.5 * std::sin(theta / 2) * aa.transpose() / theta;
        for (int i = 0; i < 3; i++)
        {
            tVector daaidaa = tVector::Zero();
            daaidaa[i] = 1.0;

            dQuaterniondAA.row(1 + i) =
                (daaidaa * theta - aa[i] * aa / theta) / (theta * theta) *
                    std::sin(theta / 2) +
                aa[i] / theta * std::cos(theta / 2) / (2 * theta) * aa;
        }
    }

    // std::cout << "diff mat = \n" << dQuaterniondAA << std::endl;
    dQuaterniondAA.col(3).setZero();
    return dQuaterniondAA;
}

/**
 * \brief           calculate d(quaternion)/d(euler_angles)
*/
tMatrixXd cMathUtil::Calc_DQuaterion_DEulerAngles(const tVector &euler_angles,
                                                  eRotationOrder order)
{
    tMatrixXd dqdeuler = tMatrixXd::Zero(4, 3);
    if (order == eRotationOrder ::XYZ)
    {
        double e_x = euler_angles[0], e_y = euler_angles[1],
               e_z = euler_angles[2];
        double cx = std::cos(e_x / 2), sx = std::sin(e_x / 2);
        double cy = std::cos(e_y / 2), sy = std::sin(e_y / 2);
        double cz = std::cos(e_z / 2), sz = std::sin(e_z / 2);
        dqdeuler.col(0) = 0.5 * tVector(cx * sy * sz - cy * cz * sx,
                                        sx * sy * sz + cx * cy * cz,
                                        cx * cy * sz - cz * sx * sy,
                                        -cx * cz * sy - cy * sx * sz);

        dqdeuler.col(1) = 0.5 * tVector(cy * sx * sz - cx * cz * sy,
                                        -cx * cy * sz - cz * sx * sy,
                                        cx * cy * cz - sx * sy * sz,
                                        -cy * cz * sx - cx * sy * sz);

        dqdeuler.col(2) = 0.5 * tVector(cz * sx * sy - cx * cy * sz,
                                        -cx * cz * sy - cy * sx * sz,
                                        cy * cz * sx - cx * sy * sz,
                                        sx * sy * sz + cx * cy * cz);
    }
    else
    {
        SIM_ERROR("invalid rotation order");
    }
    return dqdeuler;
}

void cMathUtil::TestCalc_DQuaterion_DEulerAngles()
{
    tVector euler_angles = tVector::Random();
    tQuaternion old_qua =
        cMathUtil::EulerAnglesToQuaternion(euler_angles, eRotationOrder::XYZ);
    double eps = 1e-5;
    tMatrixXd ideal_dqde = cMathUtil::Calc_DQuaterion_DEulerAngles(
        euler_angles, eRotationOrder::XYZ);
    // std::cout << "ideal_dqde = \n" << ideal_dqde << std::endl;
    for (int i = 0; i < 3; i++)
    {
        euler_angles[i] += eps;
        tQuaternion new_qua = cMathUtil::EulerAnglesToQuaternion(
            euler_angles, eRotationOrder::XYZ);
        tVector num_dqde =
            (cMathUtil::QuatToVec(new_qua) - cMathUtil::QuatToVec(old_qua)) /
            eps;
        tVector ideal_dqdei = ideal_dqde.col(i);
        tVector diff = ideal_dqdei - num_dqde;
        if (diff.norm() > 10 * eps)
        {
            std::cout
                << "[error] TestCalc_DQuaterion_DEulerAngles fail for col " << i
                << std::endl;
            std::cout << "ideal = " << ideal_dqdei.transpose() << std::endl;
            std::cout << "num = " << num_dqde.transpose() << std::endl;
            std::cout << "diff = " << diff.transpose() << std::endl;

            exit(0);
        }
        euler_angles[i] -= eps;
    }
    std::cout << "[log] TestCalc_DQuaterion_DEulerAngles succ\n";
}
void cMathUtil::TestCalc_DQuaterniontDAxisAngle()
{
    tVector aa = tVector::Random();
    aa[3] = 0;
    tQuaternion qua = cMathUtil::AxisAngleToQuaternion(aa);
    tMatrix dqua_daa = cMathUtil::Calc_DQuaternion_DAxisAngle(aa);
    double eps = 1e-5;
    for (int i = 0; i < 3; i++)
    {
        aa[i] += eps;
        tQuaternion new_qua = cMathUtil::AxisAngleToQuaternion(aa);
        tVector num_deriv_raw = (new_qua.coeffs() - qua.coeffs()) / eps;
        tVector num_deriv;
        num_deriv[0] = num_deriv_raw[3];
        num_deriv.segment(1, 3) = num_deriv_raw.segment(0, 3);
        tVector ideal_deriv = dqua_daa.col(i);
        tVector diff = ideal_deriv - num_deriv;
        if (diff.norm() > 10 * eps)
        {
            std::cout << "[error] TestDiffQuaterniontDAxisAngle fail for " << i
                      << std::endl;
            std::cout << i << " diff = " << diff.transpose() << std::endl;
            std::cout << "ideal = " << ideal_deriv.transpose() << std::endl;
            std::cout << "num = " << num_deriv.transpose() << std::endl;
        }
        aa[i] -= eps;
    }
    std::cout << "[log] TestDiffQuaterniontDAxisAngle succ\n";
}

void cMathUtil::TestCalc_Dq1q0conj_Dq0()
{
    SIM_INFO("Dq1q0conjDq0 begin test!");
    tQuaternion q1 = tQuaternion::UnitRandom(), q0 = tQuaternion::UnitRandom();
    tQuaternion old_q1_q0_conj = q1 * q0.conjugate();
    double eps = 1e-5;

    tMatrix deriv = cMathUtil::Calc_Dq1q0conj_Dq0(q0, q1);
    for (int i = 0; i < 4; i++)
    {
        switch (i)
        {
        case 0:
            q0.w() += eps;
            break;
        case 1:
            q0.x() += eps;
            break;
        case 2:
            q0.y() += eps;
            break;
        case 3:
            q0.z() += eps;
            break;

        default:
            break;
        }
        // q0.normalize();
        tQuaternion new_q1_q0_conj = q1 * q0.conjugate();
        tVector chaos_order_d =
            (new_q1_q0_conj.coeffs() - old_q1_q0_conj.coeffs()) / eps;
        tVector d = tVector(chaos_order_d[3], chaos_order_d[0],
                            chaos_order_d[1], chaos_order_d[2]);

        tVector diff = d - deriv.col(i);

        if (diff.norm() > 10 * eps)
        {
            printf("[error] TestDq1q0conjDq0_experimental fail for %d\n", i);
            std::cout << "d = " << d.transpose() << std::endl;
            // printf("d= %.5f, %.5f, %.5f, %.5f\n", );
            std::cout << "ideal d = " << deriv.col(i).transpose() << std::endl;
            std::cout << "diff = " << diff.norm() << std::endl;
            exit(0);
        }
        switch (i)
        {
        case 0:
            q0.w() -= eps;
            break;
        case 1:
            q0.x() -= eps;
            break;
        case 2:
            q0.y() -= eps;
            break;
        case 3:
            q0.z() -= eps;
            break;

        default:
            break;
        }
    }
    printf("[log] TestDq1q0conjDq0_experimental succ\n");
}

tQuaternion cMathUtil::VecDiffQuat(const tVector &v0, const tVector &v1)
{
    return tQuaternion::FromTwoVectors(v0.segment(0, 3), v1.segment(0, 3));
}

tVector cMathUtil::QuatRotVec(const tQuaternion &q, const tVector &dir)
{
    tVector rot_dir = tVector::Zero();
    rot_dir.segment(0, 3) = q * dir.segment(0, 3);
    return rot_dir;
}

tQuaternion cMathUtil::MirrorQuaternion(const tQuaternion &q, eAxis axis)
{
    tQuaternion mirror_q;
    mirror_q.w() = q.w();
    mirror_q.x() = (axis == eAxisX) ? q.x() : -q.x();
    mirror_q.y() = (axis == eAxisY) ? q.y() : -q.y();
    mirror_q.z() = (axis == eAxisZ) ? q.z() : -q.z();
    return mirror_q;
}

double cMathUtil::Sign(double val) { return SignAux<double>(val); }

int cMathUtil::Sign(int val) { return SignAux<int>(val); }

double cMathUtil::AddAverage(double avg0, int count0, double avg1, int count1)
{
    double total = count0 + count1;
    return (count0 / total) * avg0 + (count1 / total) * avg1;
}

tVector cMathUtil::AddAverage(const tVector &avg0, int count0,
                              const tVector &avg1, int count1)
{
    double total = count0 + count1;
    return (count0 / total) * avg0 + (count1 / total) * avg1;
}

void cMathUtil::AddAverage(const Eigen::VectorXd &avg0, int count0,
                           const Eigen::VectorXd &avg1, int count1,
                           Eigen::VectorXd &out_result)
{
    double total = count0 + count1;
    out_result = (count0 / total) * avg0 + (count1 / total) * avg1;
}

void cMathUtil::CalcSoftmax(const Eigen::VectorXd &vals, double temp,
                            Eigen::VectorXd &out_prob)
{
    assert(out_prob.size() == vals.size());
    int num_vals = static_cast<int>(vals.size());
    double sum = 0;
    double max_val = vals.maxCoeff();
    for (int i = 0; i < num_vals; ++i)
    {
        double val = vals[i];
        val = std::exp((val - max_val) / temp);
        out_prob[i] = val;
        sum += val;
    }

    out_prob /= sum;
}

double cMathUtil::EvalGaussian(const Eigen::VectorXd &mean,
                               const Eigen::VectorXd &covar,
                               const Eigen::VectorXd &sample)
{
    assert(mean.size() == covar.size());
    assert(sample.size() == covar.size());

    Eigen::VectorXd diff = sample - mean;
    double exp_val = diff.dot(diff.cwiseQuotient(covar));
    double likelihood = std::exp(-0.5 * exp_val);

    double partition = CalcGaussianPartition(covar);
    likelihood /= partition;
    return likelihood;
}

double cMathUtil::EvalGaussian(double mean, double covar, double sample)
{
    double diff = sample - mean;
    double exp_val = diff * diff / covar;
    double norm = 1 / std::sqrt(2 * M_PI * covar);
    double likelihood = norm * std::exp(-0.5 * exp_val);
    return likelihood;
}

double cMathUtil::CalcGaussianPartition(const Eigen::VectorXd &covar)
{
    int data_size = static_cast<int>(covar.size());
    double det = covar.prod();
    double partition = std::sqrt(std::pow(2 * M_PI, data_size) * det);
    return partition;
}

double cMathUtil::EvalGaussianLogp(const Eigen::VectorXd &mean,
                                   const Eigen::VectorXd &covar,
                                   const Eigen::VectorXd &sample)
{
    int data_size = static_cast<int>(covar.size());

    Eigen::VectorXd diff = sample - mean;
    double logp = -0.5 * diff.dot(diff.cwiseQuotient(covar));
    double det = covar.prod();
    logp += -0.5 * (data_size * std::log(2 * M_PI) + std::log(det));

    return logp;
}

double cMathUtil::EvalGaussianLogp(double mean, double covar, double sample)
{
    double diff = sample - mean;
    double logp = -0.5 * diff * diff / covar;
    logp += -0.5 * (std::log(2 * M_PI) + std::log(covar));
    return logp;
}

double cMathUtil::Sigmoid(double x) { return Sigmoid(x, 1, 0); }

double cMathUtil::Sigmoid(double x, double gamma, double bias)
{
    double exp = -gamma * (x + bias);
    double val = 1 / (1 + std::exp(exp));
    return val;
}

int cMathUtil::SampleDiscreteProb(const Eigen::VectorXd &probs)
{
    assert(std::abs(probs.sum() - 1) < 0.00001);
    double rand = RandDouble();

    int rand_idx = gInvalidIdx;
    int num_probs = static_cast<int>(probs.size());
    for (int i = 0; i < num_probs; ++i)
    {
        double curr_prob = probs[i];
        rand -= curr_prob;

        if (rand <= 0)
        {
            rand_idx = i;
            break;
        }
    }
    return rand_idx;
}

tVector cMathUtil::CalcBarycentric(const tVector &p, const tVector &a,
                                   const tVector &b, const tVector &c)
{
    tVector v0 = b - a;
    tVector v1 = c - a;
    tVector v2 = p - a;

    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0f - v - w;

    return tVector(u, v, w, 0);
}

bool cMathUtil::ContainsAABB(const tVector &pt, const tVector &aabb_min,
                             const tVector &aabb_max)
{
    bool contains = pt[0] >= aabb_min[0] && pt[1] >= aabb_min[1] &&
                    pt[2] >= aabb_min[2] && pt[0] <= aabb_max[0] &&
                    pt[1] <= aabb_max[1] && pt[2] <= aabb_max[2];
    return contains;
}

bool cMathUtil::ContainsAABB(const tVector &aabb_min0, const tVector &aabb_max0,
                             const tVector &aabb_min1, const tVector &aabb_max1)
{
    return ContainsAABB(aabb_min0, aabb_min1, aabb_max1) &&
           ContainsAABB(aabb_max0, aabb_min1, aabb_max1);
}

bool cMathUtil::ContainsAABBXZ(const tVector &pt, const tVector &aabb_min,
                               const tVector &aabb_max)
{
    bool contains = pt[0] >= aabb_min[0] && pt[2] >= aabb_min[2] &&
                    pt[0] <= aabb_max[0] && pt[2] <= aabb_max[2];
    return contains;
}

bool cMathUtil::ContainsAABBXZ(const tVector &aabb_min0,
                               const tVector &aabb_max0,
                               const tVector &aabb_min1,
                               const tVector &aabb_max1)
{
    return ContainsAABBXZ(aabb_min0, aabb_min1, aabb_max1) &&
           ContainsAABBXZ(aabb_max0, aabb_min1, aabb_max1);
}

void cMathUtil::CalcAABBIntersection(const tVector &aabb_min0,
                                     const tVector &aabb_max0,
                                     const tVector &aabb_min1,
                                     const tVector &aabb_max1, tVector &out_min,
                                     tVector &out_max)
{
    out_min = aabb_min0.cwiseMax(aabb_min1);
    out_max = aabb_max0.cwiseMin(aabb_max1);
    if (out_min[0] > out_max[0])
    {
        out_min[0] = 0;
        out_max[0] = 0;
    }
    if (out_min[1] > out_max[1])
    {
        out_min[1] = 0;
        out_max[1] = 0;
    }
    if (out_min[2] > out_max[2])
    {
        out_min[2] = 0;
        out_max[2] = 0;
    }
}

void cMathUtil::CalcAABBUnion(const tVector &aabb_min0,
                              const tVector &aabb_max0,
                              const tVector &aabb_min1,
                              const tVector &aabb_max1, tVector &out_min,
                              tVector &out_max)
{
    out_min = aabb_min0.cwiseMin(aabb_min1);
    out_max = aabb_max0.cwiseMax(aabb_max1);
}

bool cMathUtil::IntersectAABB(const tVector &aabb_min0,
                              const tVector &aabb_max0,
                              const tVector &aabb_min1,
                              const tVector &aabb_max1)
{
    tVector center0 = 0.5 * (aabb_max0 + aabb_min0);
    tVector center1 = 0.5 * (aabb_max1 + aabb_min1);
    tVector size0 = aabb_max0 - aabb_min0;
    tVector size1 = aabb_max1 - aabb_min1;
    tVector test_len = 0.5 * (size0 + size1);
    tVector delta = center1 - center0;
    bool overlap = (std::abs(delta[0]) <= test_len[0]) &&
                   (std::abs(delta[1]) <= test_len[1]) &&
                   (std::abs(delta[2]) <= test_len[2]);
    return overlap;
}

bool cMathUtil::IntersectAABBXZ(const tVector &aabb_min0,
                                const tVector &aabb_max0,
                                const tVector &aabb_min1,
                                const tVector &aabb_max1)
{
    tVector center0 = 0.5 * (aabb_max0 + aabb_min0);
    tVector center1 = 0.5 * (aabb_max1 + aabb_min1);
    tVector size0 = aabb_max0 - aabb_min0;
    tVector size1 = aabb_max1 - aabb_min1;
    tVector test_len = 0.5 * (size0 + size1);
    tVector delta = center1 - center0;
    bool overlap = (std::abs(delta[0]) <= test_len[0]) &&
                   (std::abs(delta[2]) <= test_len[2]);
    return overlap;
}

bool cMathUtil::CheckNextInterval(double delta, double curr_val,
                                  double int_size)
{
    double pad = 0.001 * delta;
    int curr_count = static_cast<int>(std::floor((curr_val + pad) / int_size));
    int prev_count =
        static_cast<int>(std::floor((curr_val + pad - delta) / int_size));
    bool new_action = (curr_count != prev_count);
    return new_action;
}

tVector cMathUtil::SampleRandPt(const tVector &bound_min,
                                const tVector &bound_max)
{
    tVector pt = tVector(RandDouble(bound_min[0], bound_max[0]),
                         RandDouble(bound_min[1], bound_max[1]),
                         RandDouble(bound_min[2], bound_max[2]), 0);
    return pt;
}

tVector cMathUtil::SampleRandPtBias(const tVector &bound_min,
                                    const tVector &bound_max)
{
    return SampleRandPtBias(bound_min, bound_max,
                            0.5 * (bound_max + bound_min));
}

tVector cMathUtil::SampleRandPtBias(const tVector &bound_min,
                                    const tVector &bound_max,
                                    const tVector &focus)
{
    double t = RandDouble(0, 1);
    tVector size = bound_max - bound_min;
    tVector new_min = focus + (t * 0.5) * size;
    tVector new_max = focus - (t * 0.5) * size;
    tVector offset = (bound_min - new_min).cwiseMax(0);
    offset += (bound_max - new_max).cwiseMin(0);
    new_min += offset;
    new_max += offset;

    return SampleRandPt(new_min, new_max);
}

void cMathUtil::QuatSwingTwistDecomposition(const tQuaternion &q,
                                            const tVector &dir,
                                            tQuaternion &out_swing,
                                            tQuaternion &out_twist)
{
    assert(std::abs(dir.norm() - 1) < 0.000001);
    assert(std::abs(q.norm() - 1) < 0.000001);

    tVector q_axis = tVector(q.x(), q.y(), q.z(), 0);
    double p = q_axis.dot(dir);
    tVector twist_axis = p * dir;
    out_twist = tQuaternion(q.w(), twist_axis[0], twist_axis[1], twist_axis[2]);
    out_twist.normalize();
    out_swing = q * out_twist.conjugate();
}

tQuaternion cMathUtil::ProjectQuat(const tQuaternion &q, const tVector &dir)
{
    assert(std::abs(dir.norm() - 1) < 0.00001);
    tVector ref_axis = tVector::Zero();
    int min_idx = 0;
    dir.cwiseAbs().minCoeff(&min_idx);
    ref_axis[min_idx] = 1;

    tVector rot_dir0 = dir.cross3(ref_axis);
    tVector rot_dir1 = cMathUtil::QuatRotVec(q, rot_dir0);
    rot_dir1 -= rot_dir1.dot(dir) * dir;

    double dir1_norm = rot_dir1.norm();
    tQuaternion p_rot = tQuaternion::Identity();
    if (dir1_norm > 0.0001)
    {
        rot_dir1 /= dir1_norm;
        p_rot = cMathUtil::VecDiffQuat(rot_dir0, rot_dir1);
    }
    return p_rot;
}

void cMathUtil::ButterworthFilter(double dt, double cutoff,
                                  Eigen::VectorXd &out_x)
{
    double sampling_rate = 1 / dt;
    int n = static_cast<int>(out_x.size());

    double wc = std::tan(cutoff * M_PI / sampling_rate);
    double k1 = std::sqrt(2) * wc;
    double k2 = wc * wc;
    double a = k2 / (1 + k1 + k2);
    double b = 2 * a;
    double c = a;
    double k3 = b / k2;
    double d = -2 * a + k3;
    double e = 1 - (2 * a) - k3;

    double xm2 = out_x[0];
    double xm1 = out_x[0];
    double ym2 = out_x[0];
    double ym1 = out_x[0];

    for (int s = 0; s < n; ++s)
    {
        double x = out_x[s];
        double y = a * x + b * xm1 + c * xm2 + d * ym1 + e * ym2;

        out_x[s] = y;
        xm2 = xm1;
        xm1 = x;
        ym2 = ym1;
        ym1 = y;
    }

    double yp2 = out_x[n - 1];
    double yp1 = out_x[n - 1];
    double zp2 = out_x[n - 1];
    double zp1 = out_x[n - 1];

    for (int t = n - 1; t >= 0; --t)
    {
        double y = out_x[t];
        double z = a * y + b * yp1 + c * yp2 + d * zp1 + e * zp2;

        out_x[t] = z;
        yp2 = yp1;
        yp1 = y;
        zp2 = zp1;
        zp1 = z;
    }
}

tMatrix cMathUtil::RotMat(const tQuaternion &quater_)
{
    // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    tMatrix res = tMatrix::Zero();
    double w = quater_.w(), x = quater_.x(), y = quater_.y(), z = quater_.z();
    res << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0,
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0,
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0, 0,
        0, 0, 1;
    return res;
}

// tQuaternion cMathUtil::RotMatToQuaternion(const tMatrix &mat)
//{
//	//
// http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
//	double eta = 0;
//	double q1, q2, q3, q4;	// = [w, x, y, z]
//
//	// determine q1
//	{
//		double detect_value = mat(0, 0) + mat(1, 1) + mat(2, 2);
//		if (detect_value > eta)
//		{
//			q1 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			double numerator = 0;
//			numerator += std::pow(mat(2, 1) - mat(1, 2), 2);
//			numerator += std::pow(mat(0, 2) - mat(2, 0), 2);
//			numerator += std::pow(mat(1, 0) - mat(0, 1), 2);
//			q1 = 0.5 *  std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	// determine q2
//	{
//		double detect_value = mat(0, 0) - mat(1, 1) - mat(2, 2);
//		if (detect_value > eta)
//		{
//			q2 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			double numerator = 0;
//			numerator += std::pow(mat(2, 1) - mat(1, 2), 2);
//			numerator += std::pow(mat(0, 1) + mat(1, 0), 2);
//			numerator += std::pow(mat(2, 0) + mat(0, 2), 2);
//			q2 = 0.5 * std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	// determine q3
//	{
//		double detect_value = -mat(0, 0) + mat(1, 1) - mat(2, 2);
//		if (detect_value > eta)
//		{
//			q3 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			double numerator = 0;
//			numerator += std::pow(mat(0, 2) - mat(2, 0), 2);
//			numerator += std::pow(mat(0, 1) + mat(1, 0), 2);
//			numerator += std::pow(mat(1, 2) + mat(2, 1), 2);
//			q3 = 0.5 * std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	// determine q4
//	{
//		double detect_value = -mat(0, 0) - mat(1, 1) + mat(2, 2);
//		if (detect_value > eta)
//		{
//			q4 = 0.5 * std::sqrt(1 + detect_value);
//		}
//		else
//		{
//			double numerator = 0;
//			numerator += std::pow(mat(1, 0) - mat(0, 1), 2);
//			numerator += std::pow(mat(2, 0) + mat(0, 2), 2);
//			numerator += std::pow(mat(2, 1) + mat(1, 2), 2);
//			q4 = 0.5 * std::sqrt(numerator / (3 - detect_value));
//		}
//	}
//
//	return tQuaternion(q1, q2, q3, q4);
//}

tVector cMathUtil::QuaternionToCoef(const tQuaternion &quater)
{
    // quaternion -> vec = [x, y, z, w]
    return tVector(quater.x(), quater.y(), quater.z(), quater.w());
}

tQuaternion cMathUtil::CoefToQuaternion(const tVector &vec)
{
    // vec = [x, y, z, w] -> quaternion
    if (vec[3] > 0)
        return tQuaternion(vec[3], vec[0], vec[1], vec[2]);
    else
        return tQuaternion(-vec[3], -vec[0], -vec[1], -vec[2]);
}

tQuaternion cMathUtil::AxisAngleToQuaternion(const tVector &angvel)
{
    double theta = angvel.norm();
    double theta_2 = theta / 2;
    double cos_theta_2 = std::cos(theta_2), sin_theta_2 = std::sin(theta_2);

    tVector norm_angvel = angvel.normalized();
    return tQuaternion(cos_theta_2, norm_angvel[0] * sin_theta_2,
                       norm_angvel[1] * sin_theta_2,
                       norm_angvel[2] * sin_theta_2);
}

// tVector cMathUtil::QuaternionToAxisAngle(const tQuaternion & quater)
//{
//	/* 	quater = [w, x, y, z]
//			w = cos(theta / 2)
//			x = ax * sin(theta/2)
//			y = ay * sin(theta/2)
//			z = az * sin(theta/2)
//		axis angle = theta * [ax, ay, az, 0]
//	*/
//	tVector axis_angle = tVector::Zero();
//
//	double theta = 2 * std::acos(quater.w());
//
//	if (theta < 1e-4) return tVector::Zero();
//
//	//std::cout << theta << " " << std::sin(theta / 2) << std::endl;
//	double ax = quater.x() / std::sin(theta / 2),
//		ay = quater.y() / std::sin(theta / 2),
//		az = quater.z() / std::sin(theta / 2);
//	return theta * tVector(ax, ay, az, 0);
//}

tVector cMathUtil::CalcAngularVelocity(const tQuaternion &old_rot,
                                       const tQuaternion &new_rot,
                                       double timestep)
{
    tQuaternion trans = new_rot * old_rot.conjugate();
    double theta = std::acos(trans.w()) * 2; // std::acos() output range [0, pi]
    if (true == std::isnan(theta))
        return tVector::Zero(); // theta = nan, when w = 1. Omega = 0, 0, 0

    if (theta > 2 * M_PI - theta)
    {
        // theta = theta - 2*pi
        theta = theta - 2 * M_PI; // -pi - pi
        trans.coeffs().segment(0, 3) *= -1;
    }
    else if (std::abs(theta) < 1e-10)
    {
        return tVector::Zero();
    }
    tVector vel = tVector::Zero();
    double coef = theta / (sin(theta / 2) * timestep);
    vel.segment(0, 3) = trans.coeffs().segment(0, 3) * coef;
    return vel;
}

tVector cMathUtil::CalcAngularVelocityFromAxisAngle(const tQuaternion &old_rot,
                                                    const tQuaternion &new_rot,
                                                    double timestep)
{
    std::cout << "cMathUtil::CalcAngularVelocityFromAxisAngle: this func "
                 "hasn't been well-tested, call another one\n";
    exit(1);
    tVector old_aa = cMathUtil::QuaternionToAxisAngle(old_rot),
            new_aa = cMathUtil::QuaternionToAxisAngle(new_rot);
    return (new_aa - old_aa) / timestep;
}

// tVector cMathUtil::QuatRotVec(const tQuaternion & quater, const tVector &
// vec)
//{
//	tVector res = tVector::Zero();
//	res.segment(0, 3) = quater * vec.segment(0, 3);
//	return res;
//}

tVector cMathUtil::QuaternionToEulerAngles(const tQuaternion &q,
                                           const eRotationOrder &order)
{
    tVector res = tVector::Zero();
    double w = q.w(), x = q.x(), y = q.y(), z = q.z();

    // handle the zero quaternion
    if (order == eRotationOrder::XYZ)
    {
        res[0] = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        res[1] = std::asin(2 * (w * y - z * x));
        res[2] = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
        // SIM_INFO("w {} x {} y {} z {}", w, x, y, z);

        // std::cout << "euler angle = " << res.transpose() << std::endl;
    }
    else if (order == eRotationOrder::ZYX)
    {
        res[0] = std::atan2(2 * (w * x - y * z), 1 - 2 * (x * x + y * y));
        res[1] = std::asin(2 * (w * y + z * x));
        res[2] = std::atan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z));
    }
    else
    {
        std::cout << "[error] tVector cMathUtil::QuaternionToEulerAngles "
                     "Unsupported rotation order = "
                  << order;
        exit(1);
    }
    return res;
}

tQuaternion cMathUtil::EulerAnglesToQuaternion(const tVector &vec,
                                               const eRotationOrder &order)
{
    tQuaternion q[3];
    for (int i = 0; i < 3; i++)
    {
        tVector axis = tVector::Zero();
        axis[i] = 1.0;

        double theta_2 = vec[i] / 2.0;
        axis = axis * std::sin(theta_2);
        axis[3] = std::cos(theta_2);

        q[i] = tQuaternion(axis[3], axis[0], axis[1], axis[2]);
    }

    tQuaternion res;
    if (order == eRotationOrder::XYZ)
    {
        res = q[2] * q[1] * q[0];
    }
    else if (order == eRotationOrder::ZYX)
    {
        res = q[0] * q[1] * q[2];
    }

    res.normalize();
    if (res.w() < 0)
        res = cMathUtil::MinusQuaternion(res);
    return res;
}

tQuaternion cMathUtil::MinusQuaternion(const tQuaternion &quad)
{
    return tQuaternion(-quad.w(), -quad.x(), -quad.y(), -quad.z());
}

tMatrix cMathUtil::EulerAnglesToRotMat(const tVector &euler,
                                       const eRotationOrder &order)
{
    // input euler angles: the rotation theta from parent to local
    // output rot mat: a rot mat that can convert a vector FROM LOCAL FRAME TO
    // PARENT FRAME
    double x = euler[0], y = euler[1], z = euler[2];
    tMatrix mat = tMatrix::Identity();
    if (order == eRotationOrder::XYZ)
    {
        tMatrix x_mat, y_mat, z_mat;
        x_mat = cMathUtil::EulerAngleRotmatX(x);
        y_mat = cMathUtil::EulerAngleRotmatY(y);
        z_mat = cMathUtil::EulerAngleRotmatZ(z);
        mat = z_mat * y_mat * x_mat;
    }
    else if (order == eRotationOrder::ZYX)
    {
        tMatrix x_mat, y_mat, z_mat;
        x_mat = cMathUtil::EulerAngleRotmatX(x);
        y_mat = cMathUtil::EulerAngleRotmatY(y);
        z_mat = cMathUtil::EulerAngleRotmatZ(z);
        mat = x_mat * y_mat * z_mat;
    }
    else
    {
        std::cout << "[error] cMathUtil::EulerAnglesToRotMat(const "
                     "tVector& euler): Unsupported rotation order"
                  << std::endl;
        exit(1);
    }
    return mat;
}

tMatrix cMathUtil::EulerAnglesToRotMatDot(const tVector &euler,
                                          const eRotationOrder &order)
{
    double x = euler[0], y = euler[1], z = euler[2];
    tMatrix mat = tMatrix::Identity();
    if (order == eRotationOrder::XYZ)
    {
        tMatrix Rz = cMathUtil::EulerAngleRotmatZ(z),
                Ry = cMathUtil::EulerAngleRotmatY(y),
                Rx = cMathUtil::EulerAngleRotmatX(x);
        tMatrix Rz_dot = cMathUtil::EulerAngleRotmatdZ(z),
                Ry_dot = cMathUtil::EulerAngleRotmatdY(y),
                Rx_dot = cMathUtil::EulerAngleRotmatdX(x);
        mat = Rz * Ry * Rx_dot + Rz_dot * Ry * Rx + Rz * Ry_dot * Rx;
    }
    else if (order == eRotationOrder::ZYX)
    {
        tMatrix Rz = EulerAngleRotmatZ(z), Ry = EulerAngleRotmatY(y),
                Rx = EulerAngleRotmatX(x);
        tMatrix Rz_dot = EulerAngleRotmatdZ(z), Ry_dot = EulerAngleRotmatdY(y),
                Rx_dot = EulerAngleRotmatdX(x);
        mat = Rx * Ry * Rz_dot + Rx_dot * Ry * Rz + Rx * Ry_dot * Rz;
    }
    else
    {
        std::cout << "[error] cMathUtil::EulerAnglesToRotMatDot(const "
                     "tVector& euler): Unsupported rotation order"
                  << std::endl;
        exit(1);
    }
    return mat;
}

tVector cMathUtil::AngularVelToqdot(const tVector &omega, const tVector &cur_q,
                                    const eRotationOrder &order)
{
    // w = Jw * q'
    // q' = (Jw)^{-1} * omega
    //[w] = R' * R^T

    // step1: get Jw
    // please read P8 formula (30) in C.K Liu's tutorial "A Quick Tutorial on
    // Multibody Dynamics" for more details
    double x = cur_q[0], y = cur_q[1], z = cur_q[2];
    tMatrix Rx = cMathUtil::EulerAngleRotmatX(x),
            Ry = cMathUtil::EulerAngleRotmatY(y),
            Rz = cMathUtil::EulerAngleRotmatZ(z);
    tMatrix Rx_dotx = cMathUtil::EulerAngleRotmatdX(x),
            Ry_doty = cMathUtil::EulerAngleRotmatdY(y),
            Rz_dotz = cMathUtil::EulerAngleRotmatdZ(z);

    if (order == eRotationOrder::XYZ)
    {
        tMatrix R = Rz * Ry * Rx;
        tMatrix dR_dx = Rz * Ry * Rx_dotx, dR_dy = Rz * Ry_doty * Rx,
                dR_dz = Rz_dotz * Ry * Rx;
        tMatrix x_col_mat = dR_dx * R.transpose(),
                y_col_mat = dR_dy * R.transpose(),
                z_col_mat = dR_dz * R.transpose();
        tVector x_col = cMathUtil::SkewMatToVector(x_col_mat);
        tVector y_col = cMathUtil::SkewMatToVector(y_col_mat);
        tVector z_col = cMathUtil::SkewMatToVector(z_col_mat);
        Eigen::Matrix3d Jw = Eigen::Matrix3d::Zero();
        Jw.block(0, 0, 3, 1) = x_col.segment(0, 3);
        Jw.block(0, 1, 3, 1) = y_col.segment(0, 3);
        Jw.block(0, 2, 3, 1) = z_col.segment(0, 3);
        tVector res = tVector::Zero();
        res.segment(0, 3) = Jw.inverse() * omega.segment(0, 3);
        return res;
    }
    else if (order == eRotationOrder::ZYX)
    {
        tMatrix R = Rx * Ry * Rz;
        tMatrix dR_dx = Rx_dotx * Ry * Rz, dR_dy = Rx * Ry_doty * Rz,
                dR_dz = Rx * Ry * Rz_dotz;
        tMatrix x_col_mat = dR_dx * R.transpose(),
                y_col_mat = dR_dy * R.transpose(),
                z_col_mat = dR_dz * R.transpose();
        tVector x_col = cMathUtil::SkewMatToVector(x_col_mat);
        tVector y_col = cMathUtil::SkewMatToVector(y_col_mat);
        tVector z_col = cMathUtil::SkewMatToVector(z_col_mat);
        Eigen::Matrix3d Jw = Eigen::Matrix3d::Zero();
        Jw.block(0, 0, 3, 1) = x_col.segment(0, 3);
        Jw.block(0, 1, 3, 1) = y_col.segment(0, 3);
        Jw.block(0, 2, 3, 1) = z_col.segment(0, 3);
        tVector res = tVector::Zero();
        res.segment(0, 3) = Jw.inverse() * omega.segment(0, 3);
        return res;
    }
    else
    {

        std::cout << "[error] cMathUtil::AngularVelToqdot: Unsupported "
                     "rotation order"
                  << std::endl;
        exit(1);
    }
}

tMatrix cMathUtil::VectorToSkewMat(const tVector &vec)
{
    tMatrix res = tMatrix::Zero();
    double a = vec[0], b = vec[1], c = vec[2];
    res(0, 1) = -c;
    res(0, 2) = b;
    res(1, 0) = c;
    res(1, 2) = -a;
    res(2, 0) = -b;
    res(2, 1) = a;

    return res;
}

tVector cMathUtil::SkewMatToVector(const tMatrix &mat)
{
    // verify mat is a skew matrix
    assert((mat + mat.transpose()).norm() < 1e-10);

    // squeeze a mat to a vector
    tVector res = tVector::Zero();
    res[0] = mat(2, 1);
    res[1] = mat(0, 2);
    res[2] = mat(1, 0);
    return res;
}

bool cMathUtil::IsSame(const tVector &v1, const tVector &v2, const double eps)
{
    for (int i = 0; i < v1.size(); i++)
        if (std::fabs(v1[i] - v2[i]) > eps)
            return false;
    return true;
}

void cMathUtil::ThresholdOp(tVectorXd &v, double threshold)
{
    v = (threshold < v.array().abs()).select(v, 0.0f);
}

double cMathUtil::Truncate(double num, int digits)
{
    return round(num * pow(10, digits)) / pow(10, digits);
}

// Nx3 friction cone
// each row is a direction now
tMatrixXd cMathUtil::ExpandFrictionCone(int num_friction_dirs,
                                        const tVector &normal_)
{
    // 1. check the input
    tVector normal = normal_;
    normal[3] = 0;
    normal.normalize();
    if (normal.norm() < 1e-6)
    {
        std::cout << "[error] ExpandFrictionCone normal = "
                  << normal_.transpose() << std::endl;
        exit(0);
    }

    // 2. generate a standard friction cone
    tMatrixXd D = tMatrixXd::Zero(4, num_friction_dirs);
    double gap = 2 * M_PI / num_friction_dirs;
    for (int i = 0; i < num_friction_dirs; i++)
    {
        D(0, i) = std::cos(gap * i);
        D(2, i) = std::sin(gap * i);
    }

    // 3. rotate the fricition cone
    tVector Y_normal = tVector(0, 1, 0, 0);
    tVector axis = Y_normal.cross3(normal).normalized();
    double theta = std::acos(Y_normal.dot(normal)); // [0, pi]
    D = cMathUtil::RotMat(cMathUtil::AxisAngleToQuaternion(axis * theta)) * D;
    D.transposeInPlace();
    // each row is a direction now
    return D;
}
tMatrix cMathUtil::InverseTransform(const tMatrix &raw_trans)
{
    std::cout << "wrong api InverseTransform should not be called\n";
    exit(1);
    tMatrix inv_trans = tMatrix::Identity();
    inv_trans.block(0, 0, 3, 3).transposeInPlace();
    inv_trans.block(0, 3, 3, 1) =
        -inv_trans.block(0, 0, 3, 3) * raw_trans.block(0, 3, 3, 1);
    return inv_trans;
}

double cMathUtil::CalcConditionNumber(const tMatrixXd &mat)
{
    Eigen::EigenSolver<tMatrixXd> solver(mat);
    tVectorXd eigen_values = solver.eigenvalues().real();
    return eigen_values.maxCoeff() / eigen_values.minCoeff();
}

/**
 * \brief		Get the jacobian preconditioner P = diag(A)
 *
 */
tMatrixXd cMathUtil::JacobPreconditioner(const tMatrixXd &A)
{
    if (A.rows() != A.cols())
    {
        std::cout << "cMathUtil::JacobPreconditioner: A is not a square matrix "
                  << A.rows() << " " << A.cols() << std::endl;
        exit(1);
    }
    tVectorXd diagonal = A.diagonal();
    if (diagonal.cwiseAbs().minCoeff() < 1e-10)
    {
        std::cout
            << "cMathUtil::JacobPreconditioner: diagnoal is nearly zero for "
            << diagonal.transpose() << std::endl;
        exit(1);
    }

    return diagonal.cwiseInverse().asDiagonal();
}

tMatrix cMathUtil::EulerAngleRotmatX(double x)
{
    tMatrix m = tMatrix::Identity();

    double cosx = cos(x);
    double sinx = sin(x);

    m(0, 0) = 1;
    m(1, 1) = cosx;
    m(1, 2) = -sinx;
    m(2, 1) = sinx;
    m(2, 2) = cosx;

    return m;
}
tMatrix cMathUtil::EulerAngleRotmatY(double y)
{
    // return AngleAxisd(y, Vector3d::UnitY()).toRotationMatrix();
    tMatrix m = tMatrix::Identity();

    double cosy = cos(y);
    double siny = sin(y);

    m(1, 1) = 1;
    m(0, 0) = cosy;
    m(0, 2) = siny;
    m(2, 0) = -siny;
    m(2, 2) = cosy;
    return m;
}
tMatrix cMathUtil::EulerAngleRotmatZ(double z)
{
    // return AngleAxisd(z, Vector3d::UnitZ()).toRotationMatrix();
    tMatrix m = tMatrix::Identity();
    m.setZero();

    double cosz = cos(z);
    double sinz = sin(z);

    m(2, 2) = 1;
    m(0, 0) = cosz;
    m(0, 1) = -sinz;
    m(1, 0) = sinz;
    m(1, 1) = cosz;

    return m;
}
tMatrix cMathUtil::EulerAngleRotmatdX(double x)
{
    tMatrix output = tMatrix::Zero();

    double cosx = cos(x);
    double sinx = sin(x);

    output(1, 1) = -sinx;
    output(1, 2) = -cosx;
    output(2, 1) = cosx;
    output(2, 2) = -sinx;
    return output;
}
tMatrix cMathUtil::EulerAngleRotmatdY(double y)
{
    tMatrix output = tMatrix::Zero();
    double cosy = cos(y);
    double siny = sin(y);

    output(0, 0) = -siny;
    output(0, 2) = cosy;
    output(2, 0) = -cosy;
    output(2, 2) = -siny;
    return output;
}
tMatrix cMathUtil::EulerAngleRotmatdZ(double z)
{
    tMatrix output = tMatrix::Zero();
    double cosz = cos(z);
    double sinz = sin(z);

    output(0, 0) = -sinz;
    output(0, 1) = -cosz;
    output(1, 0) = cosz;
    output(1, 1) = -sinz;
    return output;
}