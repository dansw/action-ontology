from action_ontologies.summarize import summarize_ontology


def test_summarize_ontology_dedupes_across_frames():
    data = {
        "video_path": "videos/example.mp4",
        "frames": [
            {
                "frame_id": "f0",
                "resources": [{"name": "left hand", "description": "empty"}],
                "entities": [{"name": "table"}],
                "actions": [{"name": "stand", "actor": "person", "target": "table"}],
            },
            {
                "frame_id": "f1",
                "resources": [
                    {"name": "Left Hand", "description": "holding egg"},
                    {"name": "egg"},
                ],
                "entities": [{"name": "table"}],
                "actions": [
                    {"name": "stand", "actor": "person", "target": "table"},
                    {"name": "hold egg", "actor": "person", "target": "egg"},
                ],
            },
        ],
    }

    summary = summarize_ontology(data)

    assert summary["video_path"] == "videos/example.mp4"
    assert summary["frame_count"] == 2
    resource_names = [item["name"] for item in summary["resources"]]
    assert resource_names == ["left hand", "egg"]
    assert summary["resources"][0]["description"] == "empty"
    entity_names = [item["name"] for item in summary["entities"]]
    assert entity_names == ["table"]
    action_names = [(item["name"], item.get("actor"), item.get("target")) for item in summary["actions"]]
    assert action_names == [
        ("stand", "person", "table"),
        ("hold egg", "person", "egg"),
    ]


def test_summarize_ontology_handles_no_frames():
    summary = summarize_ontology({"video_path": "videos/empty.mp4", "frames": []})
    assert summary["frame_count"] == 0
    assert summary["resources"] == []
    assert summary["entities"] == []
    assert summary["actions"] == []
