using System;
using System.Text;
using UnityEngine;

[Serializable]
public class MoveModel
{
    private static readonly float AGENT_SPEED = 3.0f;

    public string vertical = "";

    public string horizontal = "";

    public static MoveModel FromJsonBytes(byte[] bytes)
    {
        string json = Encoding.UTF8.GetString(bytes);
        return JsonUtility.FromJson<MoveModel>(json);
    }

    public float GetHorizontalControllerValue()
    {
        if (horizontal == "LEFT")
        {
            return -AGENT_SPEED;
        }
        if (horizontal == "RIGHT")
        {
            return AGENT_SPEED;
        }
        else return 0;
    }

    public float GetVerticalControllerValue()
    {
        if (vertical != "")
        {
            // always go forward for now
            return AGENT_SPEED;
        }
        return 0;
    }
}
