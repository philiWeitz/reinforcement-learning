using System;
using System.Text;
using UnityEngine;

[Serializable]
public class MoveModel
{
    public string vertical = "";

    public string horizontal = "";

    public static MoveModel FromJsonBytes(byte[] bytes)
    {
        string json = Encoding.UTF8.GetString(bytes);
        return JsonUtility.FromJson<MoveModel>(json);
    }
}
