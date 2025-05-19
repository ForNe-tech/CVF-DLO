using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Text;

public class CableAbstraction : MonoBehaviour
{
    public string route_name;
    public Transform parent;
    GameObject Terminal_1;
    GameObject Terminal_2;
    GameObject Cable;
    Dictionary<string, dynamic> Cable_Dict = new Dictionary<string, dynamic>();
    
    void Start()
    {
        Debug.Log("CableAbstraction_Here");
        RouteDisplayManager.Instance.GetScenePose("G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_design_DLO/ScenePose.json");
        FindChild();
        AbstractTerminalPoints();
        AbstractLinePoints();
        SaveCableDict();
    }

    public void FindChild()
    {
        foreach (Transform child in transform)
        {
            if (child.name.Substring(0, 5) == "Termi")
            {
                if (child.name.Substring(0, 10) == "Terminal_1")
                {
                    Terminal_1 = child.gameObject;
                }
                if (child.name.Substring(0, 10) == "Terminal_2")
                {
                    Terminal_2 = child.gameObject;
                }
            }
            if (child.name == "Cable")
            {
                Cable = child.gameObject;
            }
        }
    }

    public void AbstractTerminalPoints()
    {
        Vector3 Terminal_1_LocalPos = Terminal_1.transform.localPosition;
        Vector3 Terminal_1_WorldPos = this.transform.TransformPoint(Terminal_1_LocalPos);
        Vector3 Terminal_2_LocalPos = Terminal_2.transform.localPosition;
        Vector3 Terminal_2_WorldPos = this.transform.TransformPoint(Terminal_2_LocalPos);
        CreateSphere(Terminal_1_WorldPos);
        CreateSphere(Terminal_2_WorldPos);
        //float[] Terminal_1_CabinPos = ToArray(parent.InverseTransformPoint(Terminal_1_WorldPos));
        //float[] Terminal_2_CabinPos = ToArray(parent.InverseTransformPoint(Terminal_2_WorldPos));
        float[] Terminal_1_CabinPos = ToArray(Terminal_1_WorldPos);
        float[] Terminal_2_CabinPos = ToArray(Terminal_2_WorldPos);
        Cable_Dict["Terminal_1"] = Terminal_1_CabinPos;
        Cable_Dict["Terminal_2"] = Terminal_2_CabinPos;
    }

    public void AbstractLinePoints()
    {
        List<Vector3> Vertices = GetMeshVertice(Cable);
        WriteVerticesToFile(Vertices);

        int len = Vertices.Count;
        int interval = len / 100;
        Debug.Log("VerticesCount:" + len);
        for (int i = 0; i < 100; i++)
        {
            Vector3 Vertex = Vertices[i * interval];
            Vector3 VertexInWorld = Cable.transform.TransformPoint(Vertex);
            CreateSphere(VertexInWorld);
        }
    }

    public List<Vector3> GetMeshVertice(GameObject target)
    {
        MeshFilter meshFilter = target.GetComponent<MeshFilter>();
        Mesh mesh = meshFilter.mesh;
        List<Vector3> vertices = new List<Vector3>();
        mesh.GetVertices(vertices);
        return vertices;
    }

    public void WriteVerticesToFile(List<Vector3> Vertices)
    {
        List<float[]> VerticesInParent = new List<float[]>();
        foreach (Vector3 localVertex in Vertices)
        {
            Vector3 worldVertex = Cable.transform.TransformPoint(localVertex);
            //Vector3 parentVertex = parent.InverseTransformPoint(worldVertex);
            float[] parentVertexArray = ToArray(worldVertex);
            VerticesInParent.Add(parentVertexArray);
        }
        // string Cable_CabinPos = JsonConvert.SerializeObject(VerticesInParent);
        Cable_Dict["Cable"] = VerticesInParent;
        
    }

    public void SaveCableDict()
    {
        string Cable_Dict_Str = JsonConvert.SerializeObject(Cable_Dict);
        try
        {
            Debug.Log("Cable Abstraction Saved here");
            string path = Application.dataPath + "/Model/LAB_CABIN_CABLES_NEW/" + route_name + ".json";
            using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite))
            {
                fs.Seek(0, SeekOrigin.Begin);
                fs.SetLength(0);
                using (StreamWriter sw = new StreamWriter(fs, Encoding.UTF8))
                {
                    sw.WriteLine(Cable_Dict_Str);
                }
            }
        }
        catch (Exception e)
        {
            print("保存失败！" + e.Message);
        }
    }

    public void CreateSphere(Vector3 point)
    {
        Debug.Log("Create Sphere Here");
        GameObject SpacePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        SpacePoint.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        SpacePoint.transform.position = point;
    }

    public static float[] ToArray(Vector3 vector)
    {
        return new float[] { vector.x, vector.y, vector.z };
    }


}
