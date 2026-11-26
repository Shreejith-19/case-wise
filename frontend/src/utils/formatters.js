export function parseAnswerSegments(text) {
  if (!text) return [];
  const regex = /"([^"\n]{12,})"|“([^”\n]{12,})”/g;
  const segments = [];
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }
    segments.push({ type: 'quote', value: match[1] || match[2] });
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    segments.push({ type: 'text', value: text.slice(lastIndex) });
  }
  return segments;
}

export function formatDocType(type) {
  if (!type) return 'Statute';
  return type
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function buildResearchMemo(result) {
  if (!result) return '';
  const lines = ['# Legal Research Memo', '', `**Question:** ${result.question}`, ''];

  if (result.abstained) {
    lines.push('**Result:** No directly applicable statutory provision was found.');
    return lines.join('\n');
  }

  lines.push('## Answer', '', result.answer, '');
  if (result.cited?.length) {
    lines.push('## Cited Provisions', '');
    result.cited.forEach((c) => {
      lines.push(`- **${c.title}**${c.label ? `, ${c.label}` : ''}`);
    });
    lines.push('');
  }
  if (result.passages?.length) {
    lines.push('## Statutory Extracts', '');
    result.passages.forEach((p) => {
      lines.push(`### ${p.title}${p.label ? ` — ${p.label}` : ''}`);
      lines.push('');
      lines.push(p.text || '');
      lines.push('');
    });
  }
  return lines.join('\n');
}
