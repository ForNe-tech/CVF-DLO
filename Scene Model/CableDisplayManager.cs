using Newtonsoft.Json;
using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;

public class CableDisplayManager : MonoBehaviour
{
    public static CableDisplayManager Instance;

    private int lengthToRemove = 5;

    public Transform parent;

    private void Awake()
    {
        Instance = this;
    }

    void Start()
    {
        string cabledir = Application.dataPath + "/Model/LAB_CABLE_est";
        DisplayAllCables(cabledir);
    }

    public void DisplayAllCables(string cabledir)
    {
        string[] filePaths = Directory.GetFiles(cabledir);
        foreach (string filePath in filePaths)
        {
            if (filePath.EndsWith(".json"))
            {
                DisplayPathToSpace(filePath);
            }

        }
    }

    public void SaveCablePrefab(string file, GameObject Cable) {
        string prefabPath = Application.dataPath + "/Model/LAB_CABLE_est/" + file + ".prefab";
#if UNITY_EDITOR
        prefabPath = AssetDatabase.GenerateUniqueAssetPath(prefabPath);
        GameObject prefab = PrefabUtility.SaveAsPrefabAsset(Cable, prefabPath);
#endif
    }

    public void DisplayPathToSpace(string filePath)
    {
        Debug.Log("Display" + filePath);
        string Name = Path.GetFileName(filePath);
        int endIndex = Name.Length - lengthToRemove > 0 ? Name.Length - lengthToRemove : 0;
        string file = Name.Substring(0, endIndex);
        Debug.Log(file);
        GameObject PathFather = new GameObject(file);
        string jsonContent = File.ReadAllText(filePath);
        List<List<float>> path3D = JsonConvert.DeserializeObject<List<List<float>>>(jsonContent);
        Material material = new Material(Shader.Find("Standard"));
        Color randomColor = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);
        material.color = randomColor;
        foreach (List<float> path_point in path3D)
        {
            Vector3 localVertex = new Vector3(path_point[0], path_point[1], path_point[2]);
            Vector3 worldVertex = parent.transform.TransformPoint(localVertex);
            CreateSphere(worldVertex, PathFather, material);
        }
        SaveCablePrefab(file, PathFather);
    }


    public static void CreateSphere(Vector3 point, GameObject Father, Material material)
    {
        GameObject SpacePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        SpacePoint.GetComponent<Collider>().enabled = false;
        SpacePoint.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        SpacePoint.transform.position = point;
        SpacePoint.transform.SetParent(Father.transform);
        try
        {
            SpacePoint.AddComponent<MeshFilter>();
            SpacePoint.AddComponent<MeshRenderer>();
            SpacePoint.GetComponent<MeshRenderer>().material = material;
        }
        catch (Exception e)
        {
            print(e.Message);
        }

    }

    private static Dictionary<string, string> JsonToDictionary(string jsonData)
    {
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonData);
    }

    private static Vector3 StringToVector3(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        return new Vector3(float.Parse(s[0]), float.Parse(s[1]), float.Parse(s[2]));
    }

    private static Quaternion StringToQuaternion(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        Quaternion q = new Quaternion();
        q.w = float.Parse(s[0]);
        q.x = float.Parse(s[1]);
        q.y = float.Parse(s[2]);
        q.z = float.Parse(s[3]);
        return q;
    }

    public static float[] ToArray(Vector3 vector)
    {
        return new float[] { vector.x, vector.y, vector.z };
    }
}
