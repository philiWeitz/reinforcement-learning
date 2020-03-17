//using System;
//using System.Text;
//using UnityEngine;

//public class RoadPositionModel
//{
//    public Vector3 position;

//    public Quaternion rotation;

//    public RoadPositionModel(Vector3 A, Vector3 B)
//    {
//        this.position = Vector3.Lerp(A, B, 0.5f);
//        position.y = 0.0f;

//        Vector3 rotationVector = (A - B).normalized;
//        this.rotation = Quaternion.LookRotation(rotationVector) * Quaternion.Euler(0, 90, 0);
//    }
//}
