"""
scripts/audit_data_leakage.py
────────────────────────────────────────────────────────
Audit dataset for data leakage issues.

Usage:
    python scripts/audit_data_leakage.py --config configs/audit_config.yaml
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import yaml
from collections import defaultdict


class DataLeakageAuditor:
    """Auditor for detecting data leakage in dataset splits."""
    
    def __init__(self, config: dict):
        self.config = config
        self.manifest_path = config['manifest_path']
        self.split_protocol = config['split_protocol']
        self.allowed_splits = set(config.get('allowed_splits', ['train', 'val', 'test']))
        self.checks = {
            'duplicate_checksums': config.get('check_duplicate_checksums', True),
            'machine_id_isolation': config.get('check_machine_id_isolation', True),
            'temporal_overlap': config.get('check_temporal_overlap', False),
            'unknown_splits': config.get('check_unknown_splits', True),
            'unknown_machine_ids': config.get('check_unknown_machine_ids', True),
            'unknown_labels': config.get('check_unknown_labels', True),
        }
        
        self.issues = []
        self.warnings = []
        self.passed_checks = []
        
    def load_manifest(self) -> pd.DataFrame:
        """Load dataset manifest."""
        df = pd.read_csv(self.manifest_path)
        print(f"Loaded manifest with {len(df)} records")
        return df
    
    def check_unknown_values(self, df: pd.DataFrame):
        """Check for unknown split values, machine IDs, and labels."""
        print("\n=== Checking for unknown values ===")
        
        # Check unknown splits
        if self.checks['unknown_splits']:
            unknown_splits = df[~df['split'].isin(self.allowed_splits)]
            if len(unknown_splits) > 0:
                self.issues.append({
                    'check': 'unknown_splits',
                    'severity': 'critical',
                    'count': len(unknown_splits),
                    'files': unknown_splits['absolute_path'].tolist()[:10]  # First 10
                })
                print(f"  ❌ Found {len(unknown_splits)} files with unknown splits")
            else:
                self.passed_checks.append('unknown_splits')
                print(f"  ✓ All splits are valid")
        
        # Check unknown machine IDs
        if self.checks['unknown_machine_ids']:
            unknown_machine_ids = df[df['machine_id'] == 'unknown']
            if len(unknown_machine_ids) > 0:
                self.warnings.append({
                    'check': 'unknown_machine_ids',
                    'severity': 'warning',
                    'count': len(unknown_machine_ids),
                    'files': unknown_machine_ids['absolute_path'].tolist()[:10]
                })
                print(f"  ⚠ Found {len(unknown_machine_ids)} files with unknown machine IDs")
            else:
                self.passed_checks.append('unknown_machine_ids')
                print(f"  ✓ All machine IDs are known")
        
        # Check unknown labels
        if self.checks['unknown_labels']:
            unknown_labels = df[df['label'] == 'unknown']
            if len(unknown_labels) > 0:
                self.issues.append({
                    'check': 'unknown_labels',
                    'severity': 'critical',
                    'count': len(unknown_labels),
                    'files': unknown_labels['absolute_path'].tolist()[:10]
                })
                print(f"  ❌ Found {len(unknown_labels)} files with unknown labels")
            else:
                self.passed_checks.append('unknown_labels')
                print(f"  ✓ All labels are known")
    
    def check_duplicate_checksums(self, df: pd.DataFrame):
        """Check for duplicate SHA-256 checksums across different splits."""
        print("\n=== Checking for duplicate checksums ===")
        
        if not self.checks['duplicate_checksums']:
            print("  ⊘ Skipped (disabled in config)")
            return
        
        # Group by checksum
        checksum_groups = df.groupby('sha256')
        
        # Find checksums that appear in multiple splits
        duplicate_issues = []
        for checksum, group in checksum_groups:
            if len(group) > 1:
                splits = set(group['split'].unique())
                if len(splits) > 1:
                    duplicate_issues.append({
                        'sha256': checksum,
                        'splits': list(splits),
                        'count': len(group),
                        'files': group['absolute_path'].tolist()
                    })
        
        if duplicate_issues:
            self.issues.append({
                'check': 'duplicate_checksums',
                'severity': 'critical',
                'count': len(duplicate_issues),
                'details': duplicate_issues[:5]  # First 5
            })
            print(f"  ❌ Found {len(duplicate_issues)} checksum groups appearing in multiple splits")
            for issue in duplicate_issues[:3]:
                print(f"    - SHA256 {issue['sha256'][:16]}... in splits {issue['splits']}")
        else:
            self.passed_checks.append('duplicate_checksums')
            print(f"  ✓ No duplicate checksums across splits")
    
    def check_machine_id_isolation(self, df: pd.DataFrame):
        """Check machine ID isolation based on split protocol."""
        print("\n=== Checking machine ID isolation ===")
        
        if not self.checks['machine_id_isolation']:
            print("  ⊘ Skipped (disabled in config)")
            return
        
        # Get machine IDs per split
        machine_ids_by_split = df.groupby('split')['machine_id'].apply(set).to_dict()
        
        if self.split_protocol == 'machine_independent':
            # Machine IDs must be disjoint across splits
            all_machine_ids = set()
            overlapping_machine_ids = []
            
            for split, machine_ids in machine_ids_by_split.items():
                overlap = all_machine_ids & machine_ids
                if overlap:
                    overlapping_machine_ids.extend(list(overlap))
                all_machine_ids.update(machine_ids)
            
            if overlapping_machine_ids:
                self.issues.append({
                    'check': 'machine_id_isolation',
                    'severity': 'critical',
                    'protocol': 'machine_independent',
                    'count': len(set(overlapping_machine_ids)),
                    'overlapping_machine_ids': list(set(overlapping_machine_ids))
                })
                print(f"  ❌ Found {len(set(overlapping_machine_ids))} machine IDs appearing in multiple splits")
                print(f"    Overlapping IDs: {list(set(overlapping_machine_ids))[:10]}")
            else:
                self.passed_checks.append('machine_id_isolation')
                print(f"  ✓ Machine IDs are disjoint across splits (machine_independent protocol)")
        
        elif self.split_protocol == 'machine_dependent':
            # Machine IDs may repeat, but we should report the overlap
            all_machine_ids = set()
            overlapping_machine_ids = []
            
            for split, machine_ids in machine_ids_by_split.items():
                overlap = all_machine_ids & machine_ids
                if overlap:
                    overlapping_machine_ids.extend(list(overlap))
                all_machine_ids.update(machine_ids)
            
            if overlapping_machine_ids:
                self.warnings.append({
                    'check': 'machine_id_isolation',
                    'severity': 'info',
                    'protocol': 'machine_dependent',
                    'count': len(set(overlapping_machine_ids)),
                    'overlapping_machine_ids': list(set(overlapping_machine_ids))
                })
                print(f"  ℹ Found {len(set(overlapping_machine_ids))} machine IDs appearing in multiple splits")
                print(f"    (Expected for machine_dependent protocol)")
                print(f"    Overlapping IDs: {list(set(overlapping_machine_ids))[:10]}")
            else:
                self.passed_checks.append('machine_id_isolation')
                print(f"  ✓ Machine IDs are disjoint (even though protocol allows overlap)")
    
    def check_temporal_overlap(self, df: pd.DataFrame):
        """Check for temporal overlap of segments from same source recording."""
        print("\n=== Checking temporal overlap ===")
        
        if not self.checks['temporal_overlap']:
            print("  ⊘ Skipped (disabled in config)")
            return
        
        # Group by source recording
        source_groups = df.groupby('source_recording')
        
        overlap_issues = []
        for source, group in source_groups:
            splits = set(group['split'].unique())
            if len(splits) > 1:
                # Check for temporal overlap
                segments = group[['split', 'segment_start', 'segment_end']].to_dict('records')
                for i in range(len(segments)):
                    for j in range(i + 1, len(segments)):
                        seg1 = segments[i]
                        seg2 = segments[j]
                        if seg1['split'] != seg2['split']:
                            # Check overlap
                            if not (seg1['segment_end'] <= seg2['segment_start'] or 
                                    seg2['segment_end'] <= seg1['segment_start']):
                                overlap_issues.append({
                                    'source': source,
                                    'segments': [seg1, seg2]
                                })
        
        if overlap_issues:
            self.issues.append({
                'check': 'temporal_overlap',
                'severity': 'critical',
                'count': len(overlap_issues),
                'details': overlap_issues[:5]
            })
            print(f"  ❌ Found {len(overlap_issues)} temporal overlaps across splits")
        else:
            self.passed_checks.append('temporal_overlap')
            print(f"  ✓ No temporal overlap across splits")
    
    def generate_summary_stats(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics."""
        summary = {
            'total_files': len(df),
            'split_counts': df['split'].value_counts().to_dict(),
            'label_counts': df['label'].value_counts().to_dict(),
            'machine_type_counts': df['machine_type'].value_counts().to_dict(),
            'machine_id_counts': df['machine_id'].value_counts().to_dict(),
            'noise_condition_counts': df['noise_condition'].value_counts().to_dict(),
        }
        
        # Per-split breakdown
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            summary[f'split_{split}_label_counts'] = split_df['label'].value_counts().to_dict()
            summary[f'split_{split}_machine_ids'] = split_df['machine_id'].unique().tolist()
        
        return summary
    
    def run_audit(self) -> dict:
        """Run full data leakage audit."""
        print("=" * 60)
        print("DATA LEAKAGE AUDIT")
        print("=" * 60)
        
        df = self.load_manifest()
        
        # Run all checks
        self.check_unknown_values(df)
        self.check_duplicate_checksums(df)
        self.check_machine_id_isolation(df)
        self.check_temporal_overlap(df)
        
        # Generate summary
        summary = self.generate_summary_stats(df)
        
        # Compile report
        report = {
            'audit_status': 'FAILED' if self.issues else 'PASSED',
            'split_protocol': self.split_protocol,
            'summary': summary,
            'passed_checks': self.passed_checks,
            'warnings': self.warnings,
            'issues': self.issues,
            'total_issues': len(self.issues),
            'total_warnings': len(self.warnings),
            'total_passed': len(self.passed_checks)
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Audit dataset for data leakage")
    parser.add_argument("--config", default="configs/audit_config.yaml", help="Path to audit config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run audit
    auditor = DataLeakageAuditor(config)
    report = auditor.run_audit()
    
    # Save report
    report_path = Path(config['audit_report_path'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"AUDIT RESULT: {report['audit_status']}")
    print("=" * 60)
    print(f"Issues: {report['total_issues']}")
    print(f"Warnings: {report['total_warnings']}")
    print(f"Passed: {report['total_passed']}")
    print(f"\nReport saved to: {report_path}")
    
    # Exit with error code if audit failed
    if report['audit_status'] == 'FAILED':
        exit(1)


if __name__ == "__main__":
    main()
