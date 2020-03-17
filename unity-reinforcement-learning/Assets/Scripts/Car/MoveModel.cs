using System;
using System.Text;
using UnityEngine;

[Serializable]
public class MoveModel
{
    public float steering = 0.0f;

    public float acceleration = 0.0f;


    public static MoveModel FromJsonBytes(byte[] bytes)
    {
        string json = Encoding.UTF8.GetString(bytes);
        return JsonUtility.FromJson<MoveModel>(json);
    }

    public static MoveModel FromInput() 
    {
        return new MoveModel
        {
            steering = Input.GetAxis("Horizontal"),
            acceleration = Input.GetAxis("Vertical")
        };
    }

    public float GetSteering()
    {
        return Math.Max(-1, Math.Min(1, steering));
    }

    public float GetAcceleration()
    {
        return Math.Max(0, Math.Min(1, acceleration));
    }
}
