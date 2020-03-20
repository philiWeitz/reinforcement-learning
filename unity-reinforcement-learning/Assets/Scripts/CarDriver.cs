using System.Collections.Generic;
using UnityEngine;

public class CarDriver : MonoBehaviour
{
    public List<AxleInfo> axleInfos;
    public float maxMotorTorque;
    public float maxSteeringAngle;
    public float brakeTorque;
    public float decelerationForce;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    void Start()
    {
        initialPosition = this.transform.position;
        initialRotation = this.transform.rotation;
    }

    void FixedUpdate()
    {
        // reset the environment
        if (Input.GetKey(KeyCode.Space) || Environment.instance.resetEnvironment)
        {
            Reset();
            return;
        }
        if (Environment.instance.isTerminalState)
        {
            FullBreak();
            return;
        }

        CheckForOffTrack();
        CheckForFinishLine();

        if (!Environment.instance.isTerminalState)
        {
            // use either remote or user input
            if (Input.anyKey)
            {
                Drive(MoveModel.FromInput());
            }
            else
            {
                Drive(Environment.instance.networkMoveModel);
            }
        }
    }

    private void CheckForOffTrack()
    {
        if (Environment.instance.trackMeshHolder)
        {
            bool isOnTrack = false;

            // check that car is on track
            foreach (AxleInfo info in axleInfos)
            {
                info.leftWheelCollider.GetGroundHit(out WheelHit hit);
                isOnTrack |= hit.collider == null || hit.collider.name == Environment.instance.trackMeshHolder.name;
            }
            Environment.instance.isOnTrack = isOnTrack;

            if (isOnTrack)
            {
                CancelInvoke();
            }
            else if (!IsInvoking())
            {
                //Debug.Log("Off track...");
                Invoke("SetTerminalStateReached", Environment.instance.timeOffTrackBeforeTerminalInSec);
            }
        }
    }

    private void CheckForFinishLine()
    {
        if (Environment.instance.finishLine)
        {
            foreach (AxleInfo info in axleInfos)
            {
                info.leftWheelCollider.GetGroundHit(out WheelHit hit);
                if (hit.collider != null && hit.collider.name == Environment.instance.finishLine.name)
                {
                    Environment.instance.isTerminalState = true;
                    Environment.instance.isFinishReached = true;
                    return;
                }
            }
        }
    }

    private void Drive(MoveModel moveModel)
    {
        GetComponent<Rigidbody>().isKinematic = false;

        float motor = maxMotorTorque * moveModel.GetAcceleration();
        float steering = maxSteeringAngle * moveModel.GetSteering();

        // set steering and acceleration
        foreach (AxleInfo info in axleInfos)
        {
            if (info.steering)
            {
                Steering(info, steering);
            }
            if (info.motor)
            {
                Acceleration(info, motor);
            }
        }
    }

    private void Acceleration(AxleInfo axleInfo, float motor)
    {
        if (motor > 0.01f)
        {
            axleInfo.leftWheelCollider.brakeTorque = 0;
            axleInfo.rightWheelCollider.brakeTorque = 0;
            axleInfo.leftWheelCollider.motorTorque = motor;
            axleInfo.rightWheelCollider.motorTorque = motor;
        }
        else if (motor < -0.01f)
        {
            Brake(axleInfo);
        } else
        {
            Deceleration(axleInfo);
        }
    }

    private void Deceleration(AxleInfo axleInfo)
    {
        axleInfo.leftWheelCollider.brakeTorque = decelerationForce;
        axleInfo.rightWheelCollider.brakeTorque = decelerationForce;
    }

    private void Steering(AxleInfo axleInfo, float steering)
    {
        axleInfo.leftWheelCollider.steerAngle = steering;
        axleInfo.rightWheelCollider.steerAngle = steering;
    }

    private void Brake(AxleInfo axleInfo)
    {
        axleInfo.leftWheelCollider.brakeTorque = brakeTorque;
        axleInfo.rightWheelCollider.brakeTorque = brakeTorque;
    }

    private void FullBreak()
    {
        GetComponent<Rigidbody>().isKinematic = true;
    }

    private void Reset()
    {
        GetComponent<Rigidbody>().isKinematic = true;
        this.transform.rotation = initialRotation;
        this.transform.position = initialPosition;

        Environment.instance.isOnTrack = true;
        Environment.instance.isTerminalState = false;
        Environment.instance.resetEnvironment = false;
        Environment.instance.isFinishReached = false;
        Environment.instance.networkMoveModel = new MoveModel();
    }

    private void SetTerminalStateReached()
    {
        if (!Environment.instance.isOnTrack)
        {
            //Debug.Log("Terminal state reached");
            Environment.instance.isTerminalState = true;
        }
    }
}

[System.Serializable]
public class AxleInfo
{
    public WheelCollider leftWheelCollider;
    public WheelCollider rightWheelCollider;
    public bool motor;
    public bool steering;
}